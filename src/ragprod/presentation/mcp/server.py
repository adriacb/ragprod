from fastmcp import FastMCP
from ragprod.presentation.mcp.lifespan.manager import lifespan
#from fastmcp.server.auth.providers.github import GitHubProvider  # o el que elijas
#from dotenv import load_dotenv
# load_dotenv(r'C:\Users\G1A-test\Documents\personal\tests\ragprod\.env')

# print("==> client_id:", os.getenv("FASTMCP_SERVER_AUTH_GITHUB_CLIENT_ID"))
# print("==> client_secret:", os.getenv("FASTMCP_SERVER_AUTH_GITHUB_CLIENT_SECRET"))
# print("==> base_url:", os.getenv("FASTMCP_SERVER_AUTH_BASE_URL"))

# auth = GitHubProvider(
#     client_id=os.getenv("FASTMCP_SERVER_AUTH_GITHUB_CLIENT_ID"),
#     client_secret=os.getenv("FASTMCP_SERVER_AUTH_GITHUB_CLIENT_SECRET"),
#     base_url=os.getenv("FASTMCP_SERVER_AUTH_BASE_URL", "http://localhost:8000")
# )


mcp = FastMCP(
    name="RAGProd 🚀",
    version="0.0.1",
    #log_level=,
    debug=True,
    lifespan=lifespan,
    #auth=auth
)