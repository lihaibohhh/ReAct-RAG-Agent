from react_agent.tools.search import search
from react_agent.tools.excel import make_excel_table
from react_agent.tools.rag import query_internal_knowledge
from react_agent.tools.make_docx import docx_tool
from react_agent.tools.markdown import md_tool

TOOLS = [search, make_excel_table, query_internal_knowledge, docx_tool, md_tool]
