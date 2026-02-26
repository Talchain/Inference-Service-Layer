## Deployment: ISL (isl)

### Environments

| Environment | URL | Branch | Render service |
|-------------|-----|--------|----------------|
| Staging | isl-staging.onrender.com | staging | olumi-isl-staging |
| Production | isl.onrender.com | main | olumi-isl |

### Required environment variables

PYTHON_VERSION=3.12 (NOT 3.13 — NumPy 1.26.4 incompatible), ISL_API_KEY.

### Build configuration

Render auto-detects Poetry if pyproject.toml exists, but the build command must use `pip install -r requirements.txt` (not Poetry). If build fails about Poetry, check the build command in Render dashboard.

> **Note:** The repo does not maintain a `requirements.txt` in version control. If the Render build command needs one, generate it with `poetry export -f requirements.txt --output requirements.txt --without-hashes` before deploying, or configure Render's build command to run this export step.

### Deploy steps

1. `git checkout staging && git pull origin staging`
2. `bash scripts/pre-push-validate.sh`
3. `git push origin staging` (triggers Render auto-deploy)

### Post-deploy verification

`curl -s https://isl-staging.onrender.com/health | jq .`

### Known failure patterns

1. **Poetry auto-detection:** Render detects Poetry, ignores requirements.txt. Fix: set build command explicitly.
2. **Python 3.13 incompatibility:** NumPy build fails. Fix: pin Python to 3.12.
3. **Pydantic vs RequestValidator 422s:** ISL returns two different 422 formats. PLoT normalises both. If UI receives unexpected error shapes, check PLoT's normalisation, not ISL.
