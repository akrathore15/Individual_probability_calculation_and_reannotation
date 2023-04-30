def userProbability(useinferpois, candidateplaces):
    i = 0
    n = useinferpois.shape[0]
    categoryFreq = {}
    nStops = 0
    user_tfidf = []
    userProb = [{}] * 200
    while i < n:
        j = i
        while(j < n and useinferpois.iloc[j]['user'] == useinferpois.iloc[i]['user']):
            j += 1
        freq = useinferpois.iloc[i:j]['UPAPP_final'].value_counts()
        for ind, val in freq.items():
            categoryFreq[ind] = categoryFreq.get(ind,0) + val
        
        nStops += (j-i)
        i = j-1
        i += 1
        
    inverseFreq = {}
    for ind, val in categoryFreq.items():
        inverseFreq[ind] = math.log(nStops/categoryFreq.get(ind,0) ,10)
    
    i = 0
    while i < n:
        j = i
        while(j < n and useinferpois.iloc[j]['user'] == useinferpois.iloc[i]['user']):
            j += 1
        freq = useinferpois.iloc[i:j]['UPAPP_final'].value_counts()
        s = 0
        for ind, val in freq.items():
            s += val
        for ind, val in freq.items():
            freq[ind] = val / s
        
        j = i
        while(j < n and useinferpois.iloc[j]['user'] == useinferpois.iloc[i]['user']):
            user_tfidf.append(freq[useinferpois.iloc[j]['UPAPP_final']] * \
            inverseFreq[useinferpois.iloc[j]['UPAPP_final']])
            j += 1
        
        s = sum(user_tfidf[i:j])
        j = i
        categories = {}
        while(j < n and useinferpois.iloc[j]['user'] == useinferpois.iloc[i]['user']):
            categories[useinferpois.iloc[j]['UPAPP_final']] = user_tfidf[j] / s
            j += 1
        
        userProb[useinferpois.iloc[i]['user']] = categories
        i = j-1
        i += 1
        
    return userProb


    
    
    
    
    def reannotate(useinferpois, candidateplaces):
    
    userProb = userProbability(useinferpois, candidateplaces)
    
    newcp = candidateplaces.copy()
    probs = []
    for i in newcp.index:
        probs.append(userProb[newcp['user'][i]].get(newcp['label'][i], 0.0001))
    newcp['userpr'] = probs
    newcp['finalpr'] = newcp['finalpr'] * newcp['userpr']
    
    nuip = useinferpois.copy()
    n = nuip.shape[0]
    i = 0
    cnt = 0
    
    while(i < n):
        allpt = newcp[newcp['spid'] == nuip.iloc[i]['spid']].copy()
        if len(allpt) == 0:
            i += 1
            continue
        finalprs = allpt.finalpr.to_list()
        maxfinalid = finalprs.index(max(finalprs))
        maxfinalid += allpt.head(1).index
        if(i % 1000 == 0):
            print(i)
        
        poiname = allpt['poiname'][maxfinalid].values[0]
        label = allpt['label'][maxfinalid].values[0]
        cnt += (nuip.at[i,'infer_poi'] != poiname)
        nuip.at[i,'infer_poi'] = poiname
        nuip.at[i,'UPAPP_final'] = label
        i += 1
    
    change = cnt*100.0/n
    print('Updated %f%% of the total points' % change)
    return nuip
