import streamlit as st
from transformers import pipeline

token_classifier = pipeline(
    "token-classification",  # Specify the task
    model="Epsilon123/abte-restaurants-distilbert-base-uncased",
    aggregation_strategy="simple"
)

classifier = pipeline(
    "text-classification",
    model="Epsilon123/absa-restaurants-distilbert-base-uncased",

)


def main():
    st.title("Restaurant Review Sentiment Analyzer")
    st.write("Analyze sentiment for specific aspects of restaurant reviews")

    # Create text input
    user_input = st.text_area(
        "Enter your restaurant review:", "The food is terrible")

    if st.button("Analyze"):
        with st.spinner("Analyzing..."):
            try:
                # Get aspects from token classifier
                results = token_classifier(user_input)
                aspects = " ".join([result['word'] for result in results])

                # Get sentiment prediction
                prediction = classifier(f'{user_input} [SEP] {aspects}')[0]

                # Display results
                st.subheader("Analysis Results:")
                st.write("**Identified Aspects:**", aspects)
                st.write("**Sentiment:**", prediction['label'])
                st.write("**Confidence Score:**", f"{prediction['score']:.2%}")

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()
