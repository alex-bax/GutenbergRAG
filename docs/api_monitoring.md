# API Monitoring 
Monitoring of the API is done with the commonly used industry tools [Prometheus](https://prometheus.io/) and [Grafana](https://grafana.com/).\
Where Prometheus is the monitoring and metrics service, saving its observations into its (for now) local time series database.\
Grafana is the observability platform used to display the Prometheus metrics into premade/custom dashboards. 

They are deployed in Azure with the following architecture:

<img src="../imgs/API monitoring.png" alt="Diagram" height="420" >\
ℹ️ See the `prometheus` and `grafana` folders for more deployment details with `.yml`.

### Metrics used
For efficiency [Prometheus FastAPI Instrumentator](https://github.com/trallnag/prometheus-fastapi-instrumentator) is used to expose many commonly used metrics, such as counters and histograms for `http_requests_total`, `http_request_duration_seconds` and HTTP status code. 

While more could be done, I've also made 3 metrics that times the 3 different stages of the RAG process: *context retrieval, reranking, answer generation*

## Putting it all together - the custom dashboard
In order to properly showcase the metrics and the monitoring capability, I've made a "traffic simulation"/[performance test](https://learning.postman.com/docs/collections/performance-testing/performance-test-configuration/) with Postman. It makes HTTP calls to different routes on the API, like so:
<IN SERT POSTMAN SIM>






From running the simulation, it the resulting dasboard:
<img src="../imgs/grafana_dashboards_simple.png" alt="Diagram" height="400" >



:idea: 