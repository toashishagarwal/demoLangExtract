import langextract as lx
import textwrap
from dotenv import load_dotenv
import os

load_dotenv()

api_key=os.getenv("GOOGLE_API_KEY")

# 1. Define the prompt and extraction rules
prompt = textwrap.dedent("""\
    Extract sentiment with confidence level, predicted star rating, recommends product and key themes in order of appearance.
    Use exact text for extractions. Do not paraphrase or overlap entities.
    Provide meaningful attributes for each entity to add context.
    Sentiment should be positive or negative or neutral.
    Predicted star rating can be 1-5.""")

# 2. Provide a high-quality example to guide the model
examples = [
    lx.data.ExampleData(
        text="""I bought this wireless headphones last month and I'm really impressed! 
                The sound quality is excellent, especially the bass. Battery life lasts 
                about 20 hours which is perfect for my daily commute. The only downside 
                is that they're a bit heavy after wearing them for 3+ hours. Overall, 
                I'd definitely recommend these to anyone looking for good value wireless headphones""",
        extractions=[
            lx.data.Extraction(
                extraction_class="sentiment",
                extraction_text="POSITIVE",
                attributes={"confidence": "0.90"}
            ),
            lx.data.Extraction(
                extraction_class="Predicted Rating",
                extraction_text="4/5 stars"
            ),
            lx.data.Extraction(
                extraction_class="recommends product",
                extraction_text="Yes"
            ),
            lx.data.Extraction(
                extraction_class="key themes",
                extraction_text="sound quality, battery life, comfort, value"
            )
        ]
    )
]

# The input text to be processed
input_text = """
    Terrible experience with this laptop. The screen started flickering after 
    just 2 weeks of use. Customer service was unhelpful and took forever to respond. 
    The keyboard feels cheap and several keys stick. For the price I paid ($1200), 
    I expected much better quality. Save your money and look elsewhere.
    """

# Run the extraction
result = lx.extract(
    text_or_documents=input_text,
    prompt_description=prompt,
    examples=examples,
    model_id="gemini-2.5-flash",
    api_key=api_key,
)

# Save the results to a JSONL file
lx.io.save_annotated_documents([result], output_name="reviews.jsonl", output_dir=".")

# Generate the visualization from the file
html_content = lx.visualize("reviews.jsonl")
with open("reviewsvisualization.html", "w") as f:
    f.write(html_content)