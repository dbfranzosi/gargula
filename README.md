# Gargula

  Gargula is a framework for the simulation of multi-species and multi-agent interactions.

  * The individuals are called **homo-virtualis (HV)**.
  * Each HV has **genes** that determine its **traits**.
  * The relation between **genes** and **traits** is fixed *a priori* by the **phenotype** of the species, which is part of the **genetics**.
  * Each HV has a **mind** that is used to decide **actions**.
  * Mind's **memories** are based on an interaction network [1].
  * Decision policy is based on a Deep Reinforcement Learning (DRL) [2] algorithm. 
  * Learning happens during HV life span with DRL and by other HVs teachings.
  * The main technologies used are: PyTorch [3] for the DNN training, Dash/Plotly [4] for Web app visualization.

  #### Instructions
    
  * Install all requirements with conda or pip

        pip install -r requirements.txt

  * You can also use Docker: https://hub.docker.com/repository/docker/dbfranzosi/gargula
  * The simulation is controlled and visualized via a Dash/Plotly web app. Run   

        python app.py        
        
  * Open http://127.0.0.1:8050/ in your browser. 
  * Play with the parameters, create your first group of homo-virtualis and enjoy their evolution.
  * You can see more details in the doc/ folder (for the moment a beamer presentation with some results).

  #### Bibliography

  * [1] Peter W. Battaglia, Razvan Pascanu, Matthew Lai, Danilo Rezende, Koray Kavukcuoglu, Interaction Networks for Learning about Objects, Relations and Physics,  [arXiv:1612.00222], Eric A. Moreno, Olmo Cerri, Javier M. Duarte, Harvey B. Newman, Thong Q. Nguyen, Avikar Periwal, Maurizio Pierini, Aidana Serikova, Maria Spiropulu, Jean-Roch Vlimant, JEDI-net: a jet identification algorithm based on interaction networks [arXiv:1908.05318]
  * [2] Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Alex Graves, Ioannis Antonoglou, Daan Wierstra, Martin Riedmiller, Playing Atari with Deep Reinforcement Learning, arXiv:1312.5602
  * [3] https://pytorch.org/
  * [4] https://plotly.com/

  ## Coruja

  Coruja provides tools for the analysis of historical data from Gargula simulations. You can open it on the navigation bar on the top.
  
  * You can choose a saved group and the HVs on the group to see plots of some variables in the group history.
  * You can see if the HVs are evolving via natural selection or if they are learning during their lives and through generations.
  * You can also do your own analysis using the Jupyter notebook 
        ./analytics/tests.ipynb 

  ##### Author: Diogo Buarque Franzosi



  



https://user-images.githubusercontent.com/11366780/203649262-bd92f828-66a4-469c-bb24-3f82476a5b3d.mp4



