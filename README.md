# Logo_Detection
Approach to answering queries on chatbot where input is a string and one image from end user 

The expected output is count of images or "Yes/No".

Approach/Flow of System:
1. Get the text input query from chatbot
2. Pre-process it and determine nature of question (Whether count or "Yes/No") using text classification.
3. Pass this output as an input to Image classifier/Image processor
4. Get the required results and show the result to end user.
