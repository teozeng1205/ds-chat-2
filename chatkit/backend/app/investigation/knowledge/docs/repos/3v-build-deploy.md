# 3v-build-deploy

> A generic build-and-deploy toolkit of shell scripts and CloudFormation templates that 3V analytics projects consume to build Docker images, push them to ECR, generate CloudFormation nested stacks, and deploy across multiple AWS environments.

> **Note**: This document was generated from the `develop` branch. The `master` branch represents what is currently running in production and may differ.

---

## Overview

This repo is a **shared CI/CD library** — it contains no application code of its own. Individual analytics projects (e.g., `priceeye-v2`, `ds-priceeye-analytics`) check out this repo alongside their own and invoke these scripts from their `source/deploy/` directory, which must contain:

| File / Directory | Purpose |
|-----------------|---------|
| `artifacts` | Pipe-delimited manifest of all deployable components |
| `artifacts-<env>` | Optional environment-specific override manifest |
| `project-config.sh` | Project-level environment variables (names, ECR path, cluster, etc.) |
| `commonfiles/*.yaml` | CloudFormation templates (copied from this repo or overridden per-project) |
| `definitions/` | Step Function ASL definition files (`.asl.json`) |
| `resources` | Optional pipe-delimited manifest of non-artifact resources (Step Functions, etc.) |
| `dockerfiles/Dockerfile.<artifact>` | Optional per-artifact Dockerfile override |

---

## Architecture Overview

```
Developer workstation
        │
        │  runs: build.sh / deploy-all.sh
        ▼
[build-environment-config.sh]          ← detects AWS account → sets DOMAIN, REGION, DOCKER_REGISTRY
        │
        ▼
[role.sh <env>]                        ← assumes cross-account IAM role for target environment
        │
        ├──► [build.sh all <version>]
        │          │
        │          │  for each artifact in `artifacts`:
        │          │    ① downloads JAR from AWS CodeArtifact  (or copies local snapshot)
        │          │    ② calls build-docker-image.sh
        │          │         ├─ selects Dockerfile (module > artifact > type > default)
        │          │         ├─ creates ECR repo if missing
        │          │         ├─ applies ecr-repository-policy.json (cross-account pull)
        │          │         └─ docker buildx build --platform linux/arm64 --push → ECR
        │          │
        │          └──► ECR: <DOCKER_REGISTRY>/<REPO_BASE>/<artifact>:<version>
        │
        ├──► [release.sh <version> [env]]
        │          │
        │          │  reads `artifacts` (or `artifacts-<env>`)
        │          │  reads `resources` (optional)
        │          │  generates $PROJECT.yaml  ← CloudFormation nested stack master file
        │          │  calls process-resources.sh → appends Step Functions / other resources
        │          │
        │          └──► $PROJECT.yaml  (e.g., priceeye-v2.yaml)
        │
        ├──► [deploy.sh <env>]
        │          │
        │          │  uploads commonfiles/ → S3 cloudformation bucket
        │          │  uploads definitions/ → S3 cloudformation bucket (if resources file present)
        │          │  aws cloudformation deploy $PROJECT.yaml
        │          │
        │          └──► CloudFormation stack: a-$PROJECT (prefix 'a-' if name starts with digit)
        │
        └──► [lambda.sh <version> <env>]
                   │
                   │  for each lambda-image / lambda-script in `artifacts`:
                   │    aws lambda update-function-code --image-uri ...
                   │
                   └──► Lambda functions updated directly (bypasses CloudFormation)


[hotfix.sh <env>]                      ← standalone: reads current image URIs from live ECS/Lambda,
                                           replaces only the targeted artifact, generates hotfix YAML

[upload-glue-script.sh <artifact|all> <env>]
                                       ← uploads Glue job Python scripts to
                                          s3://s3-atp-3victors{env}-use1-3v-glue-etl/

[update-ecr-policies.sh]              ← re-applies ecr-repository-policy.json to all repos
                                          matching $REPO_BASE prefix

[configure-pip.sh]                    ← configures pip to use CodeArtifact as package index
```

---

## Scripts

_All scripts live in `bin/`. They are expected to be called from a consuming project's deploy directory._

---

### build-environment-config.sh

**What it does**: Detects the current AWS account via `aws sts get-caller-identity` and exports the environment variables needed by all other scripts. Currently only recognizes account `590183652635` (the 3vdev build account).

**Exports**:
| Variable | Value |
|----------|-------|
| `DOMAIN` | `atpco-3v` |
| `REGION` | `us-east-1` |
| `REPOSITORY` | `3V-ATP` |
| `DOCKER_REGISTRY` | `590183652635.dkr.ecr.us-east-1.amazonaws.com` |
| `DOCKER_ENVIRONMENT` | `3vdev` |

---

### build.sh

**Usage**: `build.sh <all|artifact-name> <version> [use-snapshots]`

**What it does**: Main build entrypoint. Reads the `artifacts` file and for each matching artifact:
1. Fetches an AWS CodeArtifact auth token.
2. Downloads the versioned JAR from CodeArtifact (`aws codeartifact get-package-version-asset`) — or copies a local `-SNAPSHOT` JAR if `snapshot` mode or the optional third argument is set.
3. Calls `build-docker-image.sh` to build and push the Docker image.

Artifacts of type `lambda`, `definition`, `yaml`, `glue`, and `api-gateway` are skipped (no Docker image needed). Types `script` and `lambda-script` are built without downloading a JAR.

**Version label rules**:
- `HOTFIX` in the label → prompts for confirmation (production risk warning)
- No hyphen in the label → prompts for confirmation (likely a release build)
- Labels without a hyphen go to `$RELEASE_PREFIX` ECR path; labels with a hyphen go to `$SNAPSHOT_PREFIX` path

---

### build-docker-image.sh

**Usage**: Called internally by `build.sh`. `build-docker-image.sh <artifact> <version> <heap> <type> <other> <args>`

**What it does**: Builds and pushes a single Docker image to ECR.

**Dockerfile selection** (in priority order):
1. A `Dockerfile` found inside the artifact's own source module directory (`$PROJECT_ROOT/**/<artifact>/Dockerfile`)
2. `dockerfiles/Dockerfile.<artifact>` in the project's deploy directory
3. `dockerfiles/Dockerfile.lambda` for `lambda-image` or `lambda-script` types
4. `dockerfiles/Dockerfile.default` for all other types
5. Falls back to `config/Dockerfile.default` or `config/Dockerfile.lambda` from this repo

**Dockerfile templating**: Before building, the script runs `sed` substitutions on the chosen Dockerfile to fill in:
- `ARTIFACT` → relative path to the downloaded JAR
- `MEMORY` → Java heap size (GB)
- `OTHER_OPTIONS` → JVM flags
- `ARGS` → runtime arguments

**ECR repo management**: Creates the repo (`aws ecr create-repository`) and applies the cross-account pull policy (`ecr-repository-policy.json`) if the repo doesn't yet exist.

**Build**: Uses `docker buildx build --platform linux/arm64 --push` (ARM64 / Graviton). If `CODEARTIFACT_AUTH_REQUIRED=true`, passes a CodeArtifact token as a Docker build secret.

---

### release.sh

**Usage**: `release.sh <version> [env]`

**What it does**: Generates `$PROJECT.yaml` — a CloudFormation master stack that references each artifact's CloudFormation template as a nested stack (`AWS::CloudFormation::Stack`).

**Artifact file selection**: If an `artifacts-<env>` file exists it is preferred; otherwise the default `artifacts` file is used.

**YAML template selection** (per artifact, in priority order):
1. `commonfiles/<artifact>.yaml` (artifact-specific override)
2. `commonfiles/<type>.yaml`
3. Built-in fallback by type:

| Artifact type | CloudFormation template |
|---------------|------------------------|
| `schedule` | `scheduledtaskv2.yaml` |
| `definition`, `queue`, `launch`, `script` | `task.yaml` |
| `lambda-image`, `lambda-script` | `lambda.yaml` |
| `api-gateway` | skipped |

If the template isn't already in `commonfiles/`, it is copied there from this repo's `config/`.

**Parameters injected into each nested stack**: `FunctionName`, `AppName`, `MemorySize`, `JavaHeap`, `Timeout`, `Cpu`, `ImageUri`, `EnvironmentName`, `EnvName`, `OriginalStackName`, `Cluster`, `ClusterArn`, `Schedule`, `Project`.

At the end, calls `process-resources.sh` to append any non-artifact resources (Step Functions, etc.).

---

### deploy.sh

**Usage**: `deploy.sh <env>`

**What it does**: Deploys the master stack YAML to the target AWS environment.

1. Assumes the cross-account role for the target environment via `role.sh`.
2. Looks up the ECS cluster ARN from `$CLUSTER_NAME`.
3. Uploads all files in `commonfiles/` to `s3://s3-atp-3victors{env}-use1-cloudformation/$PROJECT/`.
4. Runs `aws cloudformation deploy` with the master stack YAML and parameter overrides (`ClusterArn`, `Cluster`, `EnvironmentName`, `EnvName`).

**Stack naming**: If `$PROJECT` starts with a digit (e.g., `3v-...`), the stack is prefixed with `a-` to satisfy CloudFormation's naming constraint.

---

### lambda.sh

**Usage**: `lambda.sh <version> <env>`

**What it does**: Directly updates Lambda function image URIs without going through CloudFormation. For each `lambda-image` or `lambda-script` artifact in the `artifacts` file, runs:
```
aws lambda update-function-code --function-name <artifact> --image-uri <DOCKER_REGISTRY>/<REPO_BASE>/<artifact>:<version>
```
Used as the final step in `deploy-all.sh` to ensure Lambda functions pick up newly built images immediately.

---

### hotfix.sh

**Usage**: `hotfix.sh <env> <version> <artifact>`

**What it does**: Generates a hotfix CloudFormation YAML (`$PROJECT-<env>.yaml`) for emergency production fixes. For the targeted artifact it uses the new `<version>` image; for all other artifacts it reads their **currently deployed** image URIs directly from live ECS task definitions or Lambda (`aws ecs describe-task-definition`, `aws lambda get-function`). This prevents unintentional changes to non-targeted components.

---

### deploy-all.sh

**Usage**: `deploy-all.sh <environment>`

**What it does**: End-to-end deployment pipeline in four steps:

```
Step 1: build.sh all <BASE_VERSION>-SNAPSHOT
Step 2: release.sh <BASE_VERSION>-SNAPSHOT <environment>
Step 3: deploy.sh <environment>
Step 4: lambda.sh <BASE_VERSION>-SNAPSHOT <environment>
```

Automatically discovers the project root by walking up the directory tree looking for `pyproject.toml`, then `cd`s into `source/deploy/`. Extracts the base version from `pyproject.toml` (`version = "X.Y+snapshot"` → `X.Y-SNAPSHOT`). Always deploys as SNAPSHOT.

---

### role.sh

**Usage**: `role.sh <env>` — outputs `export` statements; callers do `eval $(role.sh <env>)`

**What it does**: Assumes a cross-account IAM role for the target environment and prints the temporary credentials as shell `export` commands. The `3vdev` environment is special-cased: it unsets any existing role credentials to fall back to the default profile.

**Role ARN mapping**:

| Environment | AWS Account | Role |
|-------------|-------------|------|
| `3vdev` | 590183652635 | (uses default credentials) |
| `dev` | 006402604041 | `3VDEV-Build-Deploy-Role` |
| `stg` | 252249384177 | `Stg-Build-Deploy-Role` |
| `prd` | 732267085676 | `Dev-Build-Deploy-Role` |
| `clients` | 480785847244 | `Clients-Build-Deploy-Role` |
| `3vdevds` | 891377228241 | `3VDEVDS-Build-Deploy-Role` |
| `3vgold` | 590183916591 | `3VGOLD-Build-Deploy-Role` |
| `3vprod` | 539247469204 | `3VPROD-Build-Deploy-Role` |

---

### process-resources.sh

**Usage**: Called internally by `release.sh`.

**What it does**: Reads an optional `resources` file (pipe-delimited: `resource-name|resource-type|definition-file`) and appends additional CloudFormation nested stacks to `$PROJECT.yaml` for non-artifact resources such as Step Functions. If any resources reference a definition file, uploads the entire `definitions/` directory to S3 first and retrieves the S3 version ID of the ASL file to pass as `AslVersion` parameter.

---

### upload-glue-script.sh

**Usage**: `upload-glue-script.sh <artifact|all> <env>`

**What it does**: For each `glue`-type artifact in the artifacts file, finds the corresponding Python script in the project source tree and uploads it to the Glue script bucket:
```
s3://s3-atp-3victors{env}-use1-3v-glue-etl/<artifact>.py
```
Only supports 3V-internal environments (`3vdev`, `3vdevds`, `3vgold`, `3vprod`).

---

### update-ecr-policies.sh

**Usage**: Run from a project's deploy directory.

**What it does**: Iterates over all ECR repositories whose name contains `$REPO_BASE` and re-applies the standard `ecr-repository-policy.json`. Used when new AWS accounts are added and existing repos need updated cross-account pull permissions.

---

### configure-pip.sh

**What it does**: Configures pip globally to use the 3V AWS CodeArtifact PyPI repository as the primary index, with PyPI as a fallback. Used on developer machines or CI agents when installing Python packages from private 3V packages.

---

## CloudFormation Templates (`config/`)

These are the **default** CloudFormation templates. Projects can override them by placing a custom template in their own `commonfiles/` directory.

---

### task.yaml — ECS Fargate On-Demand Task

For queue-driven or event-triggered tasks (types: `definition`, `queue`, `launch`, `script`).

**Creates**:
- `AWS::ECS::TaskDefinition` — Fargate, ARM64, `awsvpc` networking
- `AWS::IAM::Role` — full access to S3, Kinesis, SQS; ECR pull; ECS RunTask; Secrets Manager
- `AWS::Logs::LogGroup` — 7-day retention, path: `<Project>/<OriginalStackName>`

**Key parameters**:

| Parameter | Description |
|-----------|-------------|
| `AppName` | Container/task name |
| `MemorySize` | Memory in MB (default: 1024) |
| `Cpu` | CPU units (default: 1024 = 1 vCPU) |
| `JavaHeap` | JVM `-Xmx` in GB (default: 2) |
| `ImageUri` | ECR image URI |

---

### scheduledtaskv2.yaml — ECS Fargate Scheduled Task

Extends `task.yaml` with an EventBridge cron rule (types: `schedule`).

**Additional resources**:
- `AWS::Events::Rule` — named `<AppName>-task`, fires the ECS task on `Schedule`
- `AWS::IAM::Role` — EventBridge role (`ECS-<Project>-<AppName>`) with `ecs:RunTask` permission
- `AWS::EC2::SecurityGroup` — attached to the Fargate task

**Additional parameters**: `Schedule` (cron expression), `VpcId`, `Subnets`.

---

### lambda.yaml — Lambda Function (Container Image)

For Lambda functions deployed as Docker images (types: `lambda-image`, `lambda-script`).

**Creates**:
- `AWS::Lambda::Function` — ARM64, image-based, VPC-attached
- `AWS::IAM::Role` — full S3, Kinesis, Kinesis Firehose, SQS; VPC execution
- `AWS::EC2::SecurityGroup` — uses `!ImportValue mainVPC`
- `AWS::Logs::LogGroup` — 7-day retention
- `AWS::CloudWatch::Alarm` — `AlarmTimeout` — fires on `HighPriorityAlarm` SNS topic when `Duration >= Timeout * 1000ms`

**Key parameters**:

| Parameter | Description |
|-----------|-------------|
| `FunctionName` | Lambda function name |
| `MemorySize` | Memory in MB (default: 512) |
| `Timeout` | Timeout in seconds (default: 60) |
| `ImageUri` | ECR image URI |
| `Subnets` | VPC subnets for Lambda VPC config |

---

## Default Dockerfiles (`config/`)

### Dockerfile.default — Java ECS Container

```dockerfile
FROM amazoncorretto:17
ENV JAVA_HEAP="-XmxMEMORYg"
ENV JAVA_OPTIONS="--add-opens java.base/java.lang=ALL-UNNAMED OTHER_OPTIONS"
ENV ARGUMENTS=ARGS
RUN yum install -y gzip
COPY ARTIFACT /root/executable.jar
COPY download/3v.security.override /root/3v.security.override
CMD java -Djava.security.properties=/root/3v.security.override $JAVA_OPTIONS $JAVA_HEAP -jar /root/executable.jar $ARGUMENTS
```

Placeholders (`ARTIFACT`, `MEMORY`, `OTHER_OPTIONS`, `ARGS`) are substituted by `build-docker-image.sh` via `sed` before building. The `3v.security.override` file sets Java DNS cache TTL to 5 seconds.

### Dockerfile.lambda — Java Lambda Container

```dockerfile
FROM amazoncorretto:17
COPY ARTIFACT ./
ENTRYPOINT [ "java", "-cp", "./*", "com.amazonaws.services.lambda.runtime.api.client.AWSLambda" ]
CMD [ OTHER_OPTIONS ]   # handler class name
```

---

## Supporting Config Files

### ecr-repository-policy.json

Cross-account ECR pull policy applied to every repository created by `build-docker-image.sh`. Grants `ecr:BatchGetImage`, `ecr:GetDownloadUrlForLayer`, etc. to all 7 3V AWS accounts, and additionally allows the `lambda.amazonaws.com` service principal to pull images for all Lambda functions across those accounts.

**Authorized account IDs**: `006402604041` (dev), `252249384177` (stg), `480785847244` (clients), `539247469204` (3vprod), `590183652635` (3vdev build), `590183916591` (3vgold), `732267085676` (prd), `891377228241` (3vdevds).

### 3v.security.override

Java security properties file embedded in every Docker image. Sets `networkaddress.cache.ttl=5` to reduce DNS caching from Java's default (forever when a SecurityManager is present) to 5 seconds, allowing faster recovery from DNS-level infrastructure changes.

### example-project-config.sh

Template for the `project-config.sh` file that consuming projects must provide. Key variables:

| Variable | Example | Description |
|----------|---------|-------------|
| `PACKAGE_PREFIX` | `threevictors-priceeye` | Maven artifact group prefix |
| `PROJECT` | `priceeye-v2` | GitHub repo / CloudFormation stack name |
| `PROJECT_ROOT` | `$HOME/git/$PROJECT/` | Absolute path to the consuming project |
| `NAMESPACE` | `com.threevictors.aws.priceeye` | Maven namespace |
| `REPO_BASE` | `3victors/priceeyev2` | ECR repository path prefix |
| `STACK_PREFIX` | `ECS-priceeye` | CloudFormation stack name prefix |
| `CLUSTER_NAME` | `ECS-priceeye` | ECS cluster name |
| `ARTIFACT_MAPPING` | `ARTIFACT_MAPPING[blacklist-ql2]="blacklist"` | Optional name remapping |

---

## Artifacts File Format

The `artifacts` file is the manifest of all deployable components. Each non-comment, non-empty line is pipe-delimited:

```
<artifact>|<memory>|<cpu>|<heap>|<other>|<type>|<additional>|<snapshot>|<args>
```

| Field | Description |
|-------|-------------|
| `artifact` | Component name; becomes the ECR repo name suffix and CloudFormation stack resource name |
| `memory` | Memory in MB for ECS/Lambda |
| `cpu` | CPU units for ECS, or timeout (seconds) for Lambda |
| `heap` | Java JVM heap in GB (passed as `-Xmx<heap>G`) |
| `other` | Additional JVM options or Lambda handler class |
| `type` | See type table below |
| `additional` | Cron expression (for `schedule`), or source ECR image name (for `definition`) |
| `snapshot` | `snapshot` to use a locally built JAR instead of CodeArtifact |
| `args` | Additional runtime arguments |

**Type values**:

| Type | Build | Deploy template | Notes |
|------|-------|-----------------|-------|
| `schedule` | Docker build | `scheduledtaskv2.yaml` | EventBridge cron-triggered ECS task |
| `definition` | Docker build | `task.yaml` | On-demand ECS task (event/queue driven) |
| `queue` | Docker build | `task.yaml` | Queue-driven ECS task |
| `launch` | Docker build | `task.yaml` | On-demand ECS task |
| `script` | Docker build (no JAR) | `task.yaml` | Script-based ECS task |
| `lambda-image` | Docker build | `lambda.yaml` | Lambda with Java JAR image |
| `lambda-script` | Docker build (no JAR) | `lambda.yaml` | Lambda with script image |
| `lambda` | Skipped | `lambda.yaml` | Lambda (zip deployment, not Docker) |
| `glue` | Skipped | — | Glue job; script uploaded by `upload-glue-script.sh` |
| `definition` | Docker build | `task.yaml` | — |
| `yaml` | Skipped | — | Pure CloudFormation resource |
| `api-gateway` | Skipped | — | Skipped entirely |

---

## Infrastructure Summary

| Resource | Count |
|----------|-------|
| Shell scripts (`bin/`) | 13 |
| CloudFormation templates (`config/`) | 3 |
| Default Dockerfiles (`config/`) | 2 |
| Supported AWS environments | 8 |
| AWS accounts with ECR cross-account pull | 7 |
