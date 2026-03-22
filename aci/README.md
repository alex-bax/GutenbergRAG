# Azure Container Instances deployment templates

These templates split the monitoring stack into three Azure Container Instance (ACI) groups:

- `obs-ui` for Grafana
- `obs-metrics` for Prometheus
- `obs-logs` for Loki + Alloy

ACI doesn't support a local bind mount from your workstation or repo checkout the way Docker Compose does. For the log stack, these templates therefore bake `loki-azure.yml` and `alloy.alloy` into custom images and only use an in-group `emptyDir` volume for Loki's ephemeral `/loki` scratch space.

The layout matches the repo's existing monitoring split:

- Grafana reads `PROMETHEUS_URL` and `LOKI_URL` from environment variables.
- Prometheus ships as a custom image that bakes in `prometheus/prometheus.yml`.
- Loki and Alloy use the `loki-lab` configs, but `obs-logs` now bakes those files into custom images built from `loki/Dockerfile` and `alloy/Dockerfile`.

## Files

- `obs-ui.yaml` - public Grafana container group
- `obs-metrics.yaml` - public Prometheus container group
- `obs-logs.yaml` - public Loki + Alloy sidecar group with baked-in configs and an ephemeral Loki scratch volume

## Recommended image tags

Build and push images with explicit tags rather than `latest`, for example:

- `<acr-login-server>/monitoring/grafana:2026-03-22.1`
- `<acr-login-server>/monitoring/prometheus:2026-03-22.1`
- `<acr-login-server>/monitoring/loki:2026-03-22.2`
- `<acr-login-server>/monitoring/alloy:2026-03-22.2`

For Loki and Alloy, the setup now uses thin wrapper images so the ACI group can stay self-contained and avoid Azure Files for config delivery.

Example push flow:

```bash
docker build -t <acr-login-server>/monitoring/grafana:2026-03-22.1 -f grafana/Dockerfile grafana
docker push <acr-login-server>/monitoring/grafana:2026-03-22.1

docker build -t <acr-login-server>/monitoring/prometheus:2026-03-22.1 -f prometheus/Dockerfile prometheus
docker push <acr-login-server>/monitoring/prometheus:2026-03-22.1

docker build -t <acr-login-server>/monitoring/loki:2026-03-22.2 -f loki/Dockerfile .
docker push <acr-login-server>/monitoring/loki:2026-03-22.2

docker build -t <acr-login-server>/monitoring/alloy:2026-03-22.2 -f alloy/Dockerfile .
docker push <acr-login-server>/monitoring/alloy:2026-03-22.2
```

## `obs-logs` runtime layout

`obs-logs.yaml` no longer expects an Azure Files config share. Instead it:

1. ships `loki-azure.yml` in the Loki image at `/etc/loki/config.yml`
2. ships `alloy.alloy` in the Alloy image at `/etc/alloy/config.alloy`
3. mounts an `emptyDir` volume at `/loki` for Loki's active index and cache directories

Because `loki-azure.yml` uses Azure Blob Storage as the shared object store, the `/loki` mount can stay ephemeral in ACI unless you later decide to preserve the local cache directories too.

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
