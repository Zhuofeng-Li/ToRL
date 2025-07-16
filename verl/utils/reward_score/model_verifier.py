# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Adapted from https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/hendrycks_math/utils.py

from pydantic import BaseModel
from octotools.engine.openai import ChatOpenAI


llm_engine = ChatOpenAI(model="gpt-4o", enable_cache=True)

class AnswerVerification(BaseModel):
    analysis: str
    true_false: bool

def compute_score(question, response, correct_answer) -> float:      
    """
    Model eavluation for multiple choice question
    """
    query_prompt = f"""
Given a multiple-choice Question, a Model Response, and its Correct Answer, determine whether the Model's prediction is correct.

The prediction is correct only if it **exactly matches** the correct choice letter (e.g., "A", "B", "C", or "D") after necessary normalization. Follow these instructions carefully:

1. If the Model Response is a number (e.g., "2", "3", etc.), map it to the corresponding option letter based on its order in the Question (e.g., 1 → A, 2 → B, etc.).
2. Ignore irrelevant text, explanations, or format differences. Extract the core predicted answer.
3. Compare the final normalized response with the Correct Answer letter.  

Question: {question}
Model response: {response}
Correct answer: {correct_answer}

Response Format:
<analysis>: Extract the core answer and explain exact match comparison
<true_false>: Return "True" only for exact matches, otherwise "False"
    """

    verification = llm_engine(query_prompt, response_format=AnswerVerification)

    true_false = verification.true_false

    return 1.0 if true_false else 0.0


if __name__ == "__main__":
    question = "Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.\n\ntrans-cinnamaldehyde was treated with methylmagnesium bromide, forming product 1.\n\n1 was treated with pyridinium chlorochromate, forming product 2.\n\n3 was treated with (dimethyl(oxo)-l6-sulfaneylidene)methane in DMSO at elevated temperature, forming product 3.\n\nhow many carbon atoms are there in product 3?\n\nA) 11\nB) 10\nC) 12\nD) 14"
    response = "To solve the problem, we need to understand the chemical reactions involved and then determine the final structure of the compound to count the number of carbon atoms in the product.\n\n1. **Step 1: Understanding the first reaction:**\n\n   - The trans-cinnamaldehyde (C9H8O) reacts with methylmagnesium bromide (MeMgBr) to form a Grignard addition product. The Grignard reagent adds to the double bond of the trans-cinnamaldehyde, forming a new carbon-carbon bond and adding a methyl group to the structure.\n   \n   - The product of this reaction (1) is (E)-9-methylcinnamaldehyde, which has the formula C10H10O.\n\n2. **Step 2: Understanding the second reaction:**\n\n   - The product of the first reaction (1) is treated with pyridinium chlorochromate (PCC), which performs a hydroxylation reaction. This converts the aldehyde group to a hydroxyl group, forming a new alcohol product.\n   \n   - The product of this reaction (2) is a 9-methylcinnamyl alcohol, which has the formula C10H12O.\n\n3. **Step 3: Understanding the third reaction:**\n\n   - The product of the second reaction (2) is treated with (dimethyl(oxo)-16-sulfaneylidene)methane in DMSO at elevated temperature, which performs a sulfonation reaction. This converts the hydroxyl group to a sulfonic acid group, forming a new sulfonic acid product.\n   \n   - The product of this reaction (3) is a 9-methylcinnamoyl sulfonic acid, which has the formula C10H10O3S.\n\nNow, we need to count the number of carbon atoms in the final product (3). From the formula C10H10O3S, we can see that there are 10 carbon atoms.\n\nSo, the number of carbon atoms in product 3 is \\(\\boxed{10}\\).\n\nThe answer is B) 10."
    correct_answer = "B"
    res=compute_score(question, response, correct_answer)
    print(res)