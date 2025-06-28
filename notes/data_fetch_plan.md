## we love some pseudocode
## we also have to tokenize.... what a drag

if not (we have a list of works, wordcount) and not (some flag to update our list):
    we use our master list to determine which works to sample from, and count their words

what files do we have

training, datafetch, preprocessor

training:
  ## step 1
  fetch data, names of works, number of nodes in each
  i.e. make a dictionary: save that dictionary, work_index
  this entails two sparql queries, one to get all the works, one to pass that work into another query and get the COUNT
  
  pass that dictionary to the preprocessor
  preprocessor will go through each work and nab every token
  every token gets added to a dictionary, keeping track of the count (why? because it makes me happy I want to see the most common tokens)
  
  I think we're now ready for the trainig loop
  while epoch:
    do a query to get a batch, or n queries to get a batch
    this entails a new query
