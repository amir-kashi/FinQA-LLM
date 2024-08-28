# %% Imports
import os
from langchain.agents import load_tools, initialize_agent, AgentType
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


class FinQA:
    # Define a prompt template for generating questions based on provided financial data.
    prompt_template = ChatPromptTemplate.from_template(
        "You are a financial assistant. Answer the following question based on the "
        "provided information in the required format (percentage, currency, etc.).\n\n"
        "If the answer can not be determined from the information provided, please "
        "indicate this by answering 'Not Enough Context'.\n\n"
        "Please provode ONLY the final answer to the question.\n\n"
        "### Pre-Text:\n\n{pre_text}\n\n"
        "### Table:\n\n{table}\n\n"
        "### Post-Text:\n\n{post_text}\n\n"
        "### Question:\n\n{question}"
    )

    def __init__(self, gpt_model) -> None:
        """
        Initialize the FinQA instance with the specified GPT model.

        Args:
            gpt_model (str): The model name for GPT (e.g., "gpt-4o-mini").
        """
        self.gpt_model = gpt_model
        # Set up the language model client
        self.llm_client = ChatOpenAI(
            model=gpt_model,
            openai_api_key=os.environ.get("OPENAI_API_KEY"),
            temperature=0.0,  # Ensures consistent output
        )
        # Initialize the agent once during the instantiation of the class
        self.agent = self.init_agent()

    def init_agent(self):
        """
        Initialize the agent with LLM and tools.

        Returns:
            Agent: The initialized agent for handling queries.
        """
        # Load the necessary tools (like math calculations)
        tools = load_tools(["llm-math"], llm=self.llm_client)
        # Initialize the agent with the provided tools
        agent = initialize_agent(
            tools=tools,
            llm=self.llm_client,
            agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            handle_parsing_errors=True,
            verbose=True,  # Enable detailed logging
        )
        return agent

    def get_answer(
        self, pre_text: str, table: str, post_text: str, question: str
    ) -> str:
        """
        Get the answer to a financial question using the initialized agent.

        Args:
            pre_text (str): Contextual information before the table.
            table (str): The financial data in table format.
            post_text (str): Contextual information after the table.
            question (str): The question to be answered.

        Returns:
            str: The final answer from the agent.
        """
        # Create the prompt using the template
        prompt = self.prompt_template.invoke(
            {
                "pre_text": pre_text,
                "table": table,
                "post_text": post_text,
                "question": question,
            }
        )

        try:
            # Attempt to get a response from the agent
            response = self.agent(prompt.messages[0].content)
            answer = response["output"]
            return answer, "AGENT RESPONSE"
        except Exception as e:
            # Fallback to using the raw LLM client if the agent fails
            print(f"Agent failed with error: {e}. Falling back to LLM client.")
            response = self.llm_client.invoke(prompt)
            answer = response.content
            return answer, "LLM RESPONSE"
