# **Multi-Persona NLP Fine-Tuning: Character-Driven Text Generation**

## **Project Overview**

This Jupyter Notebook (multipersonaNLP.ipynb) demonstrates a complete workflow for **fine-tuning a Causal Language Model (CLM)** (specifically **DistilGPT2**) to generate text that is highly conditional on a detailed **character profile**.

The goal is to create a single model capable of role-playing multiple fictional characters, adapting its **tone, speech style, emotional undertone**, and **world role** based on the structured prompt provided. The approach uses a controlled data synthesis and cleaning pipeline combined with the Hugging Face transformers library for efficient model training.

## **Key Features**

* **Structured Data Generation:** Automatically creates a synthetic, diverse dataset of character-prompt-response pairs, ensuring balanced representation across multiple defined personas.  
* **Persona-Driven Prompts:** The training data is formatted using an **instruction-tuning** (Alpaca-style) template that explicitly embeds a detailed character profile (Name, Tone, Role, etc.) into the prompt, forcing the model to learn the association.  
* **Custom End-of-Response Token:** Utilizes a custom \[END\_OF\_RESPONSE\] token to clearly delineate where the model's desired output should terminate during training and generation.  
* **Standardized Pipeline:** Implements a clear flow for Data Preparation, Fine-Tuning, and Interactive Testing.  
* **Interactive Testing:** Includes a function to test the fine-tuned model directly, allowing the user to select a character, provide an instruction, and add optional world context.

## **Prerequisites**

To run this notebook successfully, you need a Python environment with the following libraries installed:

pip install pandas datasets transformers torch scikit-learn

## **Project Structure and Files**

| File | Description |
| :---- | :---- |
| multipersonaNLP.ipynb | The main Jupyter Notebook containing all the code for data generation, preparation, fine-tuning, and testing. |
| diverse\_characters\_dataset.csv | *(Generated)* The synthesized dataset of character responses. |
| prepared\_data/ | *(Generated Directory)* Contains the tokenized and split training and evaluation datasets saved in Hugging Face's Dataset format. |
| finetuned\_model\_output\_BASIC/ | *(Generated Directory)* The main output directory for the training process. |
| finetuned\_model\_output\_BASIC/final\_model\_cpu\_basic/ | *(Generated Directory)* The final saved fine-tuned model and tokenizer. |

## **Model Personas**

The script initializes and trains the model on the following six distinct character profiles, each defined by unique attributes:

| Character Name | Tone | Speech Style | World Role |
| :---- | :---- | :---- | :---- |
| **Aethelred** | Formal, diplomatic | Verbose, structured | Royal Advisor |
| **Ser Kaelin** | Blunt, honorable | Direct, disciplined | Knight Commander |
| **Mira the Veiled Seer** | Mysterious, poetic | Cryptic, symbolic | Oracle of the Moon Shrine |
| **Thornwick** | Sarcastic, witty | Playful, sharp | Rogue and Information Broker |
| **Elyndra** | Warm, cheerful | Soft, bright | Forest Healer |
| **Varrun** | Analytical, logical | Precise, factual | Scholar and Historian |

## **How to Run the Code**

The notebook is designed to be run sequentially from top to bottom. The final cell executes the main pipeline:

1. **Data Generation & Preparation (prepare\_data()):** Creates diverse\_characters\_dataset.csv, cleans the data (handling duplicates, length outliers), formats it into the required instruction-tuning structure, and tokenizes it, saving the result to the prepared\_data/ directory.  
2. **Model Fine-Tuning (fine\_tune\_model()):** Loads the pre-trained distilgpt2 model, prepares the Trainer arguments, and trains the model for 3 epochs, saving the final model to finetuned\_model\_output\_BASIC/final\_model\_cpu\_basic/.  
3. **Interactive Testing (test\_finetuned\_model()):** Loads the saved model and tokenizer, prompting the user for a **Character Name**, **Instruction (question)**, and **Optional Context** to demonstrate the multi-persona capability.

**Example Interactive Test Flow:**

1. **Enter the character name:** Mira the Veiled Seer  
2. **Enter your question:** What dangers lie ahead?  
3. **Enter optional world context:** A war is stirring in the northern frontier.

The model will then generate a response in Mira's **mysterious, poetic, and prophetic** style, potentially incorporating the context.

## **Technical Details**

### **Model Configuration**

| Parameter | Value | Description |
| :---- | :---- | :---- |
| **Base Model** | distilgpt2 | A smaller, faster version of GPT-2, suitable for rapid fine-tuning on CPU/GPU. |
| **Epochs** | 3 | Number of training passes over the entire dataset. |
| **Max Sequence Length** | 256 | Maximum number of tokens for input sequences. |
| **Custom EOS Token** | \[END\_OF\_RESPONSE\] | Used to ensure the model stops generating at the correct place. |

