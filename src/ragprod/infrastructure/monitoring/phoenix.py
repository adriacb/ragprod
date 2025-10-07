import phoenix as px
from phoenix.otel import register
from opentelemetry.trace import Status, StatusCode
from .base import BaseMonitor
from ragprod.core.retriever.base import BaseRetriever


class PhoenixMonitor(BaseMonitor):
    def __init__(
            self, 
            endpoint: str = "http://127.0.0.1:6006/v1/traces",
            project_name: str = "ragprod",
        ):
        self.endpoint = endpoint
        self.tracer_provider_phoenix = register(
            project_name=project_name,
            endpoint=endpoint,
        )
        self.tracer = self.get_tracer()

    def launch_app(self):
        return px.launch_app()
    
    def get_tracer(self, name: str = __name__):
        """Get the tracer for the phoenix application.
        
        ```
        @tracer.chain # used as a decorator to trace the function
        def my_function():
            pass
        ```
        """
        return self.tracer_provider_phoenix.get_tracer(name)
    
    def retrieve(self, query: str, retriever: BaseRetriever):

        with self.tracer.start_as_current_span("retrieving_documents", openinference_span_kind = "retrieveer") as span:
            span.add_event("Starting retrieval")
            span.set_input(query)

            try:
                documents = retriever.retrieve(query)
                span.set_attribute("retrieval.documents.count", len(documents))
                for idx, document in enumerate(documents):
                    span.set_attribute(f"retrieval.documents.{idx}.document_id", idx)
                    span.set_attribute(f"retrieval.documents.{idx}.document_content", document.content)
                    span.set_attribute(f"retrieval.documents.{idx}.document_metadata", document.metadata)

            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.set_attribute("error.type", type(e).__name__)
                span.set_attribute("error.message", str(e))
                raise e
            finally:
                span.set_status(Status(StatusCode.OK, "Retrieval completed"))
                return documents
