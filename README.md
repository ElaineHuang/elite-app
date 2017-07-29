# cf-sample-app-python

Push a simple Python Flask App to Cloud Foundry

## Prerequisites

* Cloud Foundry account
* [Cloud Foundry CLI](https://github.com/cloudfoundry/cli#downloads)
* [Git](https://git-scm.com/downloads)

## 1. Log into Cloud Foundry
```
cf login --skip-ssl-validation
```

## 2. Clone the sample App

```
git clone https://github.com/rickymax/cf-sample-app-python
cd cf-sample-app-python
```

## 3. Modify manifest to fit your requirements (optional)

Manifests provide consistency and reproducibility, and can help you automate deployment, especially of multiple applications at once. [Learn more.](https://docs.cloudfoundry.org/devguide/deploy-apps/manifest.html "Deploying with Application Manifests")

```
---
applications:
- name: tic100-sample-app
  random-route: true
  instances: 1
  memory: 128M
  disk_quota: 512M
```

## 4. Push the App
```
cf push
```

