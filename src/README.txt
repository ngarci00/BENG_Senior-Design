To test out polygons or rectangles, or any tweaks to the overall model please go into .config
- Here uncomment what's needed and comment was not, anything with the name rec = rectangles, poly = polygons etc

TO CHECK GO IN THIS ORDER:    ⸜(｡˃ ᵕ ˂ )⸝♡ 
1 - build_index.py 
    make sure the index.json file ise the one we want, also the script gives us an insight into the distribution of the data.
2- make_split.py 
    - use --help when running in the terminal to see arguments
    example: python src/**/make_splits.py --subset-per-class #
        in our case # is 5 we want 5 PASS , 5 FAIL = 10 videos to quick train the model.
    - ouput should be: splits_name.json 
3 - dataset.py 
    - shows us a sample from a random video that went through the clip tensor for the model,  so returns: 
     x, y, video_id
     x: is tensor shaped (T,C,H,W)

     T = clip length
     C = 3
     H,W = resise_hw which is (112,112)
4 - run.py
    main script this will strat the learning on the model. PLEASE make sure other parameters are done before reaching here!
5 - eval.py 
    statistical analysis script run it after training the model!
    returns figs, tables, conf matrix, roc curve -> for each fold and compares them too!
6 - biomarker_eval.py
    statistical analysis scrypt, precense rate for each biomarker and how acc improves or declines when present, how often they were..
     recognized in the model etc. 
7 - once finalized you can change the reports/ dir name to anything needed. please keep the same name until this step to avoid errors
//// MOST IMPORTANTLY PLEASE CHECK THE INPUT_DIR AND OUTPUT_DIR -> in your terminal the path prints out pay attention to !!! /////