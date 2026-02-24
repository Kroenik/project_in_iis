from agent import get_agent
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver
from custom_classes import Context


def main():

    model = init_chat_model(
        "gpt-4o-mini", temperature=0.5, timeout=10, max_tokens=1000
    )
    checkpointer = InMemorySaver()
    config = {"configurable": {"thread_id": "1"}}
    context = Context(user_id=2)
    agent = get_agent(model, checkpointer, Context)
    while True:
        # some_input: str = input("Enter some input (or 'exit' to quit):")
        some_input = input("Enter some input:")
        if some_input == "exit":
            break
        print("Thinking...")
        agent_input = {"messages": [{"role": "user", "content": some_input}]}

        result = agent.invoke(agent_input, config=config, context=context)
        print(result["structured_response"])


if __name__ == "__main__":
    main()
