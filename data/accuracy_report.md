# Accuracy Report

In this project, two models have been used for generating answers to financial questions:

## Models Overview

1. **Agent Based**

   The **Math Agent** is a specialized LLM with a focus on mathematical operations. Upon receiving a prompt, it reformulates the question into a mathematical operation by asking itself clarifying questions. It then searches for answers in the provided context and performs the necessary calculations.

   However, the Math Agent may not always provide an answer, particularly in cases where the question requires judgment or descriptive responses rather than mathematical operations.

2. **Standard LLM**

   When the Math Agent is unable to provide an answer, the prompt is passed to a standard LLM model. This model handles questions that do not involve mathematical calculations but may require general language understanding.

   Both models receive the same prompt format:
   
   ```
   You are a financial assistant. Answer the following question based on the provided information in the required format (percentage, currency, etc.).
   
   If the answer cannot be determined from the information provided, please indicate this by answering 'Not Enough Context'.
   
   Please provide ONLY the final answer to the question.
   
   ### Pre-Text:
   {pre_text}
   
   ### Table:
   {table}
   
   ### Post-Text:
   {post_text}
   
   ### Question:
   {question}
   ```

## Error Analysis

Examining the cases where the models generated incorrect answers reveals several common issues:

1. **Misinterpreted Calculations**

   The models sometimes misinterpret the question, calculating either the percentage change instead of absolute change or vice versa, due to unclear question phrasing.

2. **Incorrect Original Data**

   In some cases, the models have provided correct calculations based on the question, but the original data was incorrect.

3. **Different Representations**

   The models sometimes return answers in different formats compared to the original data (e.g., "120 million" vs. "120,000,000").

## Suggestions for Improvement

To enhance the accuracy and reliability of the models, I would suggest the following improvements:

- **Add More Agents**: Incorporate additional agents, such as a Python agent, capable of writing and executing code to handle more complex queries.
  
- **Multi-Agent Structure**: Implement a multi-agent system to leverage the strengths of various agents for different types of questions.

- **Output Reformatting**: Introduce an additional layer to reformat the output to match the question’s requirements more precisely.

These enhancements can help address the current limitations and improve overall performance in generating accurate responses.
