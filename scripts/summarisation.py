from transformers import pipeline

summarizer = pipeline("summarization", model = "sshleifer/distilbart-cnn-12-6")

def summarise_text(txt):
  """
    
    Uses pretrained DistilBART summarisation model from Huggingface Ecosystem to summarise long texts, by reducing them into chunks of
    maximum number of words = 600, passing text through model, and concatenating separate chunks together.
    
    Args:
        1. raw text (str)

    Returns: 
        1. summarised text (str)

  """

  max_chunk = 600 #max: 600 words per chunk
  text = txt.replace('.', '.<eos>')
  text = text.replace('?', '?<eos>')
  text = text.replace('!', '!<eos>')

  sentences = text.split('<eos>')
  current_chunk = 0 
  chunks = []
  # Because summarizer function has a limit on the size of input text that it takes, we have to split the text into smaller chunks
  for sentence in sentences:
      if len(chunks) == current_chunk + 1: 
          if len(chunks[current_chunk]) + len(sentence.split(' ')) <= max_chunk: #if current chunk still has space for another sentence
              chunks[current_chunk].extend(sentence.split(' '))
          else:
              current_chunk += 1
              chunks.append(sentence.split(' ')) #move onto next chunk, and add current chunk to chunks list
      else:
          chunks.append(sentence.split(' ')) 

  for chunk_id in range(len(chunks)):
      chunks[chunk_id] = ' '.join(chunks[chunk_id])

  res = []
  for chunk_num in range(len(chunks)):
    if len(chunks[chunk_num].split(' ')) < 120: #for chunks of text with less than 120 words, use entire chunk rather than summarising
      temp_res = summarizer(chunks[chunk_num], max_length = len(chunks[chunk_num].split(' ')), min_length=10, do_sample=False)
    else: #summarise these chunks
      temp_res = summarizer(chunks[chunk_num], max_length = 120, min_length=10, do_sample=False) #max & min num of words per summary
    res.append(temp_res[0]['summary_text']) 
  output_txt = ' '.join(res) #join summarised chunks together
  return output_txt
