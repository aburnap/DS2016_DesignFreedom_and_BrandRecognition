#-----------------------------------------------------------------------------
#               Compute the Full Rank List from Partial Rank by Pagerank
#               Author: Yanxin Pan
#               Date: Dec 4, 2014
#               Notes: This is the code used for rank aggregation of attributes
#               pagerank.py will be import
#-----------------------------------------------------------------------------


from pagerank import compare_two_ranks 

attr_list = ['Active', 'Aggressive','Distinctive','Expressive','Innovative','Luxurious','Powerful','Sporty','Well Proportioned','Youthful']

#Two Markov Chain generation function (adj_m) is available. 'adj_m1' and 'adj_m2'
for attr_name in attr_list:
	compare_two_ranks(attr_name,52,0.8, 'adj_m1')