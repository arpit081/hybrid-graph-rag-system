import json
import os
import requests
import streamlit as st

CHATBOT_URL = os.getenv("CHATBOT_URL", "http://localhost:8000/hospital-rag-agent")

with st.sidebar:
    st.header("About")
    st.markdown(
        """
        This chatbot interfaces with a
        [LangChain](https://python.langchain.com/docs/get_started/introduction)
        agent designed to answer questions about the hospitals, patients,
        visits, physicians, and insurance payers in  a fake hospital system.
        The agent uses  retrieval-augment generation (RAG) over both
        structured and unstructured data that has been synthetically generated.
        """
    )

    st.header("Example Questions")
    st.markdown("- Which hospitals are in the hospital system?")
    st.markdown("""- What is the current wait time at wallace-hamilton hospital?""")
    st.markdown(
        """- At which hospitals are patients complaining about billing and
        insurance issues?"""
    )
    st.markdown("- What is the average duration in days for closed emergency visits?")
    st.markdown(
        """- What are patients saying about the nursing staff at
        Castaneda-Hardy?"""
    )
    st.markdown("- What was the total billing amount charged to each payer for 2023?")
    st.markdown("- What is the average billing amount for medicaid visits?")
    st.markdown("- Which physician has the lowest average visit duration in days?")
    st.markdown("- How much was billed for patient 789's stay?")
    st.markdown(
        """- Which state had the largest percent increase in medicaid visits
        from 2022 to 2023?"""
    )
    st.markdown("- What is the average billing amount per day for Aetna patients?")
    st.markdown(
        """- How many reviews have been written from
                patients in Florida?"""
    )
    st.markdown(
        """- For visits that are not missing chief complaints,
       what percentage have reviews?"""
    )
    st.markdown(
        """- What is the percentage of visits that have reviews for
        each hospital?"""
    )
    st.markdown(
        """- Which physician has received the most reviews for this visits
        they've attended?"""
    )
    st.markdown("- What is the ID for physician James Cooper?")
    st.markdown(
        """- List every review for visits treated by physician 270.
        Don't leave any out."""
    )


st.title("Hospital System Chatbot")
st.info(
    """Ask me questions about patients, visits, insurance payers, hospitals,
    physicians, reviews, and wait times!"""
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if "output" in message.keys():
            st.markdown(message["output"])

        if "explanation" in message.keys():
            with st.status("How was this generated", state="complete"):
                st.info(message["explanation"])

if prompt := st.chat_input("What do you want to know?"):
    st.chat_message("user").markdown(prompt)

    st.session_state.messages.append({"role": "user", "output": prompt})

    data = {"text": prompt, "stream": True}

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        explanation = []

        try:
            response = requests.post(
                CHATBOT_URL,
                json=data,
                headers={"Accept": "text/event-stream"},
                stream=True,
                timeout=120,
            )

            if response.status_code == 200:
                content_type = response.headers.get("content-type", "")
                if "text/event-stream" in content_type:
                    for line in response.iter_lines():
                        if not line:
                            continue
                        line_text = (
                            line.decode("utf-8") if isinstance(line, bytes) else line
                        )
                        if line_text.startswith("data: "):
                            try:
                                payload = json.loads(line_text[6:])
                            except Exception:
                                continue
                            event_type = payload.get("type")
                            if event_type == "token":
                                full_response += payload.get("token", "")
                                message_placeholder.markdown(full_response + "▌")
                            elif event_type == "done":
                                full_response = payload.get("output", full_response)
                                explanation = payload.get("intermediate_steps", [])
                    message_placeholder.markdown(full_response)
                else:
                    # Non-streaming JSON fallback
                    json_data = response.json()
                    full_response = json_data.get("output", "")
                    explanation = json_data.get("intermediate_steps", [])
                    message_placeholder.markdown(full_response)
            else:
                full_response = """An error occurred while processing your message.
This usually means the chatbot failed at generating a query to
answer your question. Please try again or rephrase your message."""
                explanation = [full_response]
                message_placeholder.markdown(full_response)
        except Exception as e:
            full_response = f"Connection error: {e}"
            explanation = [full_response]
            message_placeholder.markdown(full_response)

    if explanation:
        with st.status("How was this generated?", state="complete"):
            st.info(explanation)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "output": full_response,
            "explanation": explanation,
        }
    )
