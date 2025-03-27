import logging
logging.basicConfig(filename="monitoring/inference.log", level=logging.INFO)

def run_inference(query, doc_gpt):
    logging.info(f"Query: {query}")
    response = doc_gpt.run(query)
    logging.info(f"Response: {response}")
    return response