# Azure Container Instances deployment templates

These templates split the monitoring stack into three Azure Container Instance (ACI) groups:

- `obs-ui` for Grafana
- `obs-metrics` for Prometheus
- `obs-logs` for Loki + Alloy

The layout matches the repo's existing monitoring split:

- Grafana reads `PROMETHEUS_URL` and `LOKI_URL` from environment variables.
- Prometheus ships as a custom image that bakes in `prometheus/prometheus.yml`.
- Loki and Alloy use the `loki-lab` configs, but `obs-logs` mounts those configs from Azure Files instead of baking them into the image.

## Files

- `obs-ui.yaml` - public Grafana container group
- `obs-metrics.yaml` - public Prometheus container group
- `obs-logs.yaml` - public Loki + Alloy sidecar group with Azure Files config mounts

## Recommended image tags

Build and push images with explicit tags rather than `latest`, for example:

- `<acr-login-server>/monitoring/grafana:2026-03-22.1`
- `<acr-login-server>/monitoring/prometheus:2026-03-22.1`
- `<acr-login-server>/monitoring/loki:2026-03-22.1`
- `<acr-login-server>/monitoring/alloy:2026-03-22.1`

For Loki and Alloy, you can either build your own thin wrapper images or simply mirror pinned upstream images into ACR under the same pinned tags. Because `obs-logs` mounts the configs from Azure Files, the images only need the binaries, not baked-in config.

Example push flow:

```bash
docker build -t <acr-login-server>/monitoring/grafana:2026-03-22.1 -f grafana/Dockerfile grafana
docker push <acr-login-server>/monitoring/grafana:2026-03-22.1

docker build -t <acr-login-server>/monitoring/prometheus:2026-03-22.1 -f prometheus/Dockerfile prometheus
docker push <acr-login-server>/monitoring/prometheus:2026-03-22.1

docker pull grafana/loki:2.9.8
docker tag grafana/loki:2.9.8 <acr-login-server>/monitoring/loki:2026-03-22.1
docker push <acr-login-server>/monitoring/loki:2026-03-22.1

docker pull grafana/alloy:<pinned-alloy-version>
docker tag grafana/alloy:<pinned-alloy-version> <acr-login-server>/monitoring/alloy:2026-03-22.1
docker push <acr-login-server>/monitoring/alloy:2026-03-22.1
```

## `obs-logs` volume layout

`obs-logs.yaml` expects two Azure Files shares:

1. A config share mounted read-only at `/mnt/config`
2. A Loki state share mounted at `/loki`

Place these files into the config share root:

- `loki-azure.yml`
- `alloy.alloy`

Example upload commands:

```bash
az storage file upload \
  --account-name <storage-account-name> \
  --account-key '<storage-account-key>' \
  --share-name <azure-files-share-with-loki-and-alloy-config> \
  --source loki-lab/config/loki-azure.yml \
  --path loki-azure.yml

az storage file upload \
  --account-name <storage-account-name> \
  --account-key '<storage-account-key>' \
  --share-name <azure-files-share-with-loki-and-alloy-config> \
  --source loki-lab/config/alloy.alloy \
  --path alloy.alloy
```

## Deployment order

Deploy in this order so Grafana can point at live backends:

1. `obs-logs`
2. `obs-metrics`
3. `obs-ui`

## Deploy commands

```bash
az container create --resource-group <resource-group> --file aci/obs-logs.yaml
az container create --resource-group <resource-group> --file aci/obs-metrics.yaml
az container create --resource-group <resource-group> --file aci/obs-ui.yaml
```

## Notes

- All three templates use public IPs and DNS labels for simplicity. Move them into a VNet later if you want private-only access.
- `obs-logs` uses `http://localhost:3100/loki/api/v1/push` so Alloy can push to Loki over the shared container-group network.
- Replace every `<placeholder>` before deployment.
- If you prefer managed identity for ACR pulls later, swap the `imageRegistryCredentials` block accordingly.
