def compute_probs(neg,pos):
  p0 = neg/(neg+pos)
  p1 = pos/(neg+pos)
  return [p0,p1]

def cond_prob(full_table, e, e_val, t, t_val):
  t_subset = up_table_subset(full_table, t, 'equals', t_val)
  e_list = up_get_column(t_subset, e)
  p_b_a = sum([1 if v==e_val else 0 for v in e_list])/len(e_list)        
  return p_b_a

def cond_probs_product(table, e_val, t_col, t_val):
  cond_prob_list = []
  for pair in evidence_complete:
    xi = pair[0]
    yi = pair[1]
    cond_prob_list += [cond_prob(flu_table_2, xi, yi, target_column, target_val)]
  partial_numerator = up_product(cond_prob_list) 
  return partial_numerator

def prior_prob(table, t_col, t_val):
  t_list = up_get_column(table, t_col)
  p_a = sum([1 if v==t_val else 0 for v in t_list])/len(t_list)
  return p_a



def test_it():
  return 'loaded'
