# Graph Attention Recurrent Neural Network for Bilateral Trade Prediction and Adjacency Matrix Sampling
Graph Neural Networks (GNN) are a powerful technique to model data on non-eucledian domains with neural network universal function approximator. They are mainly used on static networks where nodes and edges do not change over time. To overcome this issue new models extended the GNN model to incorporate temporal data and the resulting model is defined as a Dynamic Graph Neural Networks (DGNN). We use this technique to model the bilateral trade evolution of the International Trade Network (ITN) where nodes in the network represent the countries, encoded as a feature vector and the edges represent the trade relations between two countries. We analyze the topological and statistical properties of the estimated model and visualize the evolution of relations between countries. We then evaluate the model predictive performance on link prediction and reconstruction capabilities. <br/>
![alt-text](https://github.com/claCase/Master-Thesis/blob/master/Results/temporal_adj.gif)
![alt-text](https://github.com/claCase/Master-Thesis/blob/master/Results/temporal_distr.gif)
![alt-text](https://github.com/claCase/Master-Thesis/blob/master/Results/parameter_uncertainty.png)
![alt-text](https://github.com/claCase/Master-Thesis/blob/master/Results/in_out_clos.png)
![alt-text](https://github.com/claCase/Master-Thesis/blob/master/Results/final_emb.png)
![alt-text](https://github.com/claCase/Master-Thesis/blob/master/Results/jaccard3.png)
