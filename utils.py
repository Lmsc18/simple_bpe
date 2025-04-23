def merge(ids,pair,idx):
    new_ids=[]
    i=0
    while i<len(ids):
        if ids[i]==pair[0] and i<len(ids)-1 and ids[i+1]==pair[1]:
            new_ids.append(idx)
            i+=2
        else:
            new_ids.append(ids[i])
            i+=1
    return new_ids

def getstat(ids:list[int]):
    counts={}
    for pairs in zip(ids,ids[1:]):
        counts[pairs]=counts.get(pairs,0)+1
    return counts

