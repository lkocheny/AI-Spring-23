def compute_probs(neg,pos):
  p0 = neg/(neg+pos)
  p1 = pos/(neg+pos)
  return [p0,p1]

def cond_prob(full_table, e_col, e_val, t_col, t_val):
  t_subset = up_table_subset(full_table, t_col, 'equals', t_val)
  e_list = up_get_column(t_subset, e_col)
  p_b_a = sum([1 if v==e_val else 0 for v in e_list])/len(e_list)        
  return p_b_a + .01  #Laplace smoothing factor

def cond_probs_product(table, e_val, t_col, t_val):
  cond_prob_list = []
  table_columns = up_list_column_names(table)
  evidence_columns = table_columns[:-1]
  evidence_columns
  evidence_complete = up_zip_lists(evidence_columns, e_val)
  for pair in evidence_complete:
    xi = pair[0]
    yi = pair[1]
    cond_prob_list += [cond_prob(table, xi, yi, t_col, t_val)]
  partial_numerator = up_product(cond_prob_list) 
  return partial_numerator

def prior_prob(table, t_col, t_val):
  t_list = up_get_column(table, t_col)
  p_a = sum([1 if v==t_val else 0 for v in t_list])/len(t_list)
  return p_a

def naive_bayes(table, evidence_row, target):
  #compute P(Flu=0|...) by collecting cond_probs in a list, take the product of the list, finally multiply by P(Flu=0)
  p_num = cond_probs_product(table, evidence_row, target, 0)
  p_a = prior_prob(table, target, 0)
  neg = p_num * p_a
  #do same for P(Flu=1|...)
  p_num = cond_probs_product(table, evidence_row, target, 1)
  p_a = prior_prob(table, target, 1)
  pos = p_num * p_a
  #Use compute_probs to get 2 probabilities
  neg,pos = compute_probs(neg,pos)
  #return your 2 results in a list
  return [neg,pos]

def metrics(inputlist):
  assert isinstance(inputlist,list), f'Parameter must be a list'
  for item in inputlist:
    assert isinstance(item,list), f'Parameter must be a list of lists'
    assert len(item) == 2, f'Parameter must be a zipped list'
    for value in item:
      assert isinstance(value, int), f'All values in the pair must be an integer'
      assert value>=0, f'All values in the pair must be greater or equal to 0'
  tn = sum([1 if pair==[0,0] else 0 for pair in inputlist])
  tp = sum([1 if pair==[1,1] else 0 for pair in inputlist])
  fp = sum([1 if pair==[1,0] else 0 for pair in inputlist])
  fn = sum([1 if pair==[0,1] else 0 for pair in inputlist])
  accuracy =  sum([p==a for p,a in inputlist])/len(inputlist)
  precision = tp/(tp+fp) if tp+fp>0 else 0
  recall = tp/(tp+fn) if tp+fn>0 else 0
  f1 = 2*precision*recall/(precision+recall) if precision+recall>0 else 0
  return {'Precision': {precision}, 'Recall': {recall}, 'F1': {f1}, 'Accuracy': {accuracy}}

def test_it():
  return 'loaded'
