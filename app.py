import streamlit as st
from transformers import GPT2Tokenizer, BertTokenizer, AutoTokenizer
import transformers

# Directory for DeepSeek tokenizer (if you have a custom model there)
chat_tokenizer_dir = "deepseek/"

def tokenize_text(text, tokenizer_type):
    """
    Tokenizes the given text using the selected tokenizer.
    Returns a tuple (tokens, token_ids).
    """
    if tokenizer_type == "GPT-2":
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokens = tokenizer.tokenize(text)
        token_ids = tokenizer.convert_tokens_to_ids(tokens)

    elif tokenizer_type == "BERT":
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        tokens = tokenizer.tokenize(text)
        token_ids = tokenizer.convert_tokens_to_ids(tokens)

    elif tokenizer_type == "DeepSeek":
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            chat_tokenizer_dir, trust_remote_code=True
        )
        tokens = tokenizer.tokenize(text)
        token_ids = tokenizer.encode(text)

    elif tokenizer_type in ("GPT-4", "GPT-4o"):
        try:
            import tiktoken
        except ImportError:
            st.error("The tiktoken package is required for GPT-4 tokenization. Please install it via `pip install tiktoken`.")
            return [], []
        # Use tiktoken's encoding for GPT-4
        encoding = tiktoken.encoding_for_model(tokenizer_type.lower())
        token_ids = encoding.encode(text)
        # Since tiktoken returns token IDs, we can get a rough token representation by decoding each individual token ID.
        tokens = [encoding.decode([tid]) for tid in token_ids]

    else:
        tokens, token_ids = [], []

    return tokens, token_ids


def visualize_tokens(tokens):
    """
    Creates an HTML snippet to show tokens in colored boxes.
    """
    html = "<div style='display: flex; flex-wrap: wrap;'>"
    # List of colors for token boxes.
    colors = ["#e6194B", "#3cb44b", "#ffe119", "#4363d8",
              "#f58231", "#911eb4", "#46f0f0", "#f032e6",
              "#bcf60c", "#fabebe"]
    for i, token in enumerate(tokens):
        color = colors[i % len(colors)]
        # Each token is displayed in a colored box with some margin and padding.
        html += (
            f"<div style='margin:5px; padding:5px; background-color:{color}; "
            f"border-radius:5px;'>{token}</div>"
        )
    html += "</div>"
    return html


def main():
    st.title("AI Tokenizer Visualization")
    st.write("Enter text below to see how different AI tokenizers break it into tokens.")

    # Add new tokenizer options
    tokenizer_options = ["GPT-2", "BERT", "DeepSeek", "GPT-4", "GPT-4o", "BLT (coming soon)"]
    selected_tokenizer = st.selectbox("Select a tokenizer", tokenizer_options, key="tokenizer_select")

    # Reset tokenization results when the tokenizer selection changes.
    if "current_tokenizer" not in st.session_state or st.session_state.current_tokenizer != selected_tokenizer:
        st.session_state.current_tokenizer = selected_tokenizer
        st.session_state.token_data = None
        st.session_state.token_html = None

    text_input = st.text_area(
        "Enter your text:",
        "",
        key="text_input"
    )

    if selected_tokenizer.startswith("BLT"):
        st.info("BLT is coming soon. Please select another tokenizer.")
    else:
        if st.button("Tokenize"):
            tokens, token_ids = tokenize_text(text_input, selected_tokenizer)
            if not tokens:
                st.error("No tokens generated. Please check your input or the tokenizer settings.")
            else:
                st.session_state.token_data = [{"Token": tok, "ID": tid} for tok, tid in zip(tokens, token_ids)]
                st.session_state.token_html = visualize_tokens(tokens)

        if st.session_state.get("token_data"):
            st.subheader("Tokens and Token IDs")
            st.table(st.session_state.token_data)

            st.subheader("Token Visualization")
            st.markdown(st.session_state.token_html, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
