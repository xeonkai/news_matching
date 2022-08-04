from transformers import pipeline

summarizer = pipeline("summarization", model = "sshleifer/distilbart-cnn-12-6")

def summarise_text(txt):
  max_chunk = 800 #max: 800 words per chunk
  txt = txt.replace('.', '.<eos>')
  txt = txt.replace('?', '?<eos>')
  txt = txt.replace('!', '!<eos>')

  sentences = txt.split('<eos>')
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
      
  res = summarizer(chunks, max_length=120, min_length=10, do_sample=False) #max & min num of words per summary
  output_txt = ' '.join([summ['summary_text'] for summ in res])
  return output_txt