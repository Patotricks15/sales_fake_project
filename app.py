from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langgraph.prebuilt import create_react_agent
from langchain.agents import Tool, initialize_agent
from langchain.chains import LLMMathChain, LLMChain
from langchain_community.tools import DuckDuckGoSearchResults
from sqlalchemy import create_engine
from dotenv import load_dotenv
from typing import Annotated
import operator

load_dotenv()

duckduckgo_tool = DuckDuckGoSearchResults()

# Define the state type
class State(TypedDict):
    question:  Annotated[list, operator.add]
    sql_output: Annotated[str, operator.add]
    pre_answer: Annotated[str, operator.add]
    final_output: str


# Create an SQLAlchemy engine instance for DuckDB
engine = create_engine('duckdb:///sales_fake_project/seeds/project.db')

# Initialize SQLDatabase with view_support enabled
sql_database = SQLDatabase(engine=engine, view_support=True)
# Instantiate the LLM
model_3_5 = ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0)
model_4o = ChatOpenAI(model="gpt-4o-mini", temperature=0)
# Create the SQL toolkit and get its tools
toolkit = SQLDatabaseToolkit(db=sql_database, llm=model_3_5)
sql_tools = toolkit.get_tools()


problem_chain = LLMMathChain.from_llm(llm=model_3_5)
math_tool = Tool.from_function(name="Calculator",
                func=problem_chain.run,
                description="Useful for when you need to answer questions  about math. This tool is only for math questions and nothing else. Only input math expressions.")




# Define a system prompt for the SQLAgent
sql_prefix = (
    "You are a SQLAgent specialized in interacting with a SQLite database. "
    "Given a question, generate a syntactically correct SQLite query that returns relevant information "
    "from the following tables:\n\n"
    "fact_sales (id_sale, id_customer, id_product, id_store, id_date, value, quantity) -> Contains sales transactions linking customers, products, stores, and dates.\n"
    "dim_customer (id_customer, name, gender, birth_date, city, state, country) -> Represents customer details and demographics.\n"
    "dim_product (id_product, product_name, category, brand, price) -> Stores product information including category, brand, and price.\n"
    "dim_date (id_date, full_date, day, month, year, quarter, day_of_week) -> Represents the date dimension for time analysis.\n"
    "dim_store (id_store, store_name, address, city, state, country) -> Contains store information and location details.\n"
    "vw_sales_complete (id_sale, full_date, customer_name, product_name, category, store_name, value, quantity) -> Provides a detailed view of sales by joining the fact table with all dimension tables.\n"
    "vw_monthly_sales (year, month, total_sales, total_products_sold) -> Aggregates sales data on a monthly basis.\n"
    "vw_product_ranking (product_name, total_sales, total_quantity_sold) -> Ranks products based on overall sales performance.\n"
    "vw_store_performance (store_name, total_sales, total_products_sold) -> Summarizes sales performance for each store.\n"
    "vw_customer_ranking (customer_name, total_spent, number_of_sales) -> Ranks customers based on their total spending and number of transactions.\n\n"
    "Do not perform any DML statements. Return the query results as a concise text output."
)

sql_system_message = SystemMessage(content=sql_prefix)
# Create the SQL agent
sql_agent = create_react_agent(model_3_5, sql_tools, messages_modifier=sql_system_message)


# Create the PricingAnalystAgent
pricing_prefix = (
    "You are a PricingAnalystAgent, an expert in microeconomics, pricing analysis and strategy, and have a knowledge about cannibalization effect. "
    "Given a question and context from a SQL query, provide a clear, concise final answer with insights and recommendations "
    "regarding pricing strategy. Do not generate SQL queries here; just analyze the provided context."
)
pricing_system_message = SystemMessage(content=pricing_prefix)
# Create the Pricing Analyst agent (no extra tools needed)
pricing_agent = create_react_agent(model_3_5, tools=[math_tool], messages_modifier=pricing_system_message)


# Create the PricingAnalystAgent
sales_prefix = (
    "You are a SalesAnalystAgent, an expert in business, sales analysis and strategy. "
    "Given a question and context from a SQL query, provide a clear, concise final answer with insights and recommendations "
    "regarding pricing strategy. Do not generate SQL queries here; just analyze the provided context."
)
sales_system_message = SystemMessage(content=sales_prefix)
# Create the Pricing Analyst agent (no extra tools needed)
sales_agent = create_react_agent(model_3_5, tools=[math_tool], messages_modifier=pricing_system_message)



# Create the LeadDataAnalystAgent
lead_data_prefix = (
    "You are a LeadDataAnalystAgent, an expert in data analysis, task prioritization, and strategic planning. "
    "Your role is to review the answers obtained from data queries and generate tasks that yield actionable insights. "
    "Analyze these responses, assess their feasibility, and recommend the most viable and impactful tasks to pursue. "
    "Focus solely on analyzing the provided context and recommending prioritized tasks—do not generate SQL queries here."
)
lead_data_system_message = SystemMessage(content=lead_data_prefix)
# Create the Lead Data Analyst agent (no extra tools needed)
lead_data_agent = create_react_agent(model_3_5, tools=[math_tool], messages_modifier=lead_data_system_message)




# Build the State Graph

# The state graph has two nodes: "SQLAgent" and "PricingAnalyst".
# The flow is: START -> SQLAgent -> PricingAnalyst -> END
builder = StateGraph(State)

# Node for SQLAgent: takes the input question, invokes the SQL agent, and stores its output as 'sql_output'.
builder.add_node("SQLAgent", lambda state: {
    "sql_output": sql_agent.invoke({"messages": [HumanMessage(content=state["question"])]})
})

# Node for SalesAnalystAgent: takes the original question and the SQL agent's output, and produces the final answer.
builder.add_node("SalesAnalystAgent", lambda state: {
    "pre_answer": sales_agent.invoke({
        "messages": [HumanMessage(content=f"Question: {state['question']}\nSQL Output: {state['sql_output']}")]
    })
})


# Node for PricingAnalystAgent: takes the original question and the SQL agent's output, and produces the final answer.
builder.add_node("PricingAnalystAgent", lambda state: {
    "pre_answer": pricing_agent.invoke({
        "messages": [HumanMessage(content=f"Question: {state['question']}\nSQL Output: {state['sql_output']}")]
    })
})

# Node for SalesAnalystAgent: takes the original question and the SQL agent's output, and produces the final answer.
builder.add_node("LeadDataAnalystAgent", lambda state: {
    "final_output": lead_data_agent.invoke({
        "messages": [HumanMessage(content=f"Question: {state['question']}\Final Output: {state['final_output']}")]
    })
})

builder.add_edge(START, "SQLAgent")
builder.add_edge("SQLAgent", "SalesAnalystAgent")
builder.add_edge("SQLAgent", "PricingAnalystAgent")
builder.add_edge("SalesAnalystAgent", END)
builder.add_edge("PricingAnalystAgent", END)


# Compile the state graph
graph = builder.compile()

png_bytes = graph.get_graph(xray=1).draw_mermaid_png()

# Save the PNG data to a file
with open("sales_project_graph.png", "wb") as f:
    f.write(png_bytes)

# REPL loop to ask questions and get final answers
while True:
    user_question = input("Enter your question: ")
    initial_state: State = {"question": user_question, "sql_output": "", "final_output": ""}
    final_state = graph.invoke(initial_state)
    print("Final Answer:", final_state["final_output"]['messages'][-1].content)
    print("----")