#######################################################################################################################################
## app.py
##
## The following code contains the chatbot's visualization logic using Streamlit UI
##
## This project is the exclusive property of Charles Schwab.  
## Unauthorized use, reproduction, or distribution of this project without explicit permission is strictly prohibited.
## 
#######################################################################################################################################

import streamlit as st
from business_logic import find_relevant_article, get_openai_response, detect_prompt_injection, shorten_text, is_valid_query

st.markdown(
    "<h1 style='text-align: center; font-size: 32px;'>📈 <span style='background-color: #39bfef; color: white; padding: 5px 10px; border-radius: 5px;'><i>Charles</i> SCHWAB Stock News <i>Chatbot</i></span> </h1>",
    unsafe_allow_html=True
)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display chat history
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
user_query = st.chat_input("Ask about recent stock news...")

if user_query:
    if not is_valid_query(user_query):
        st.warning("⚠️ You entered an invalid or meaningless query. Please try again with valid text.")
    else:
        # Detect prompt injection before processing
        injection_alert = detect_prompt_injection(user_query)
        
        if injection_alert:
            # Display alert and do not proceed
            with st.chat_message("assistant"):
                st.markdown(injection_alert)
            st.session_state["messages"].append({"role": "assistant", "content": injection_alert})
        
        else:
            # Display user message
            st.session_state["messages"].append({"role": "user", "content": user_query})
            with st.chat_message("user"):
                st.markdown(user_query)

            # Retrieve the best matching article
            best_match_article = find_relevant_article(user_query)

            # Get OpenAI response
            ai_response = get_openai_response(user_query, best_match_article)

            # Shorten full article text
            shortened_text = shorten_text(best_match_article.page_content, max_length=300)

            # Display assistant response
            with st.chat_message("assistant"):
                if any(phrase in ai_response.lower() for phrase in ["do not contain", "do not mention any information", "i'm sorry", "the article does not mention", "no information related"]):
                    response_content = ai_response
                    st.markdown(response_content)
                else:
                    response_content = (
                        f"**Answer:**\n\n{ai_response}\n\n"
                        f"**Search Result:**\n\n**{best_match_article.metadata.get('title', 'No Title Available')}** \n\n"
                        f"{shortened_text} [Read more...]({best_match_article.metadata.get('source', '#')})\n\n"
                        f"{':heavy_minus_sign:' * 35}"
                    )
                    st.markdown(response_content)

            # Store response in chat history
            st.session_state["messages"].append({"role": "assistant", "content": response_content})
