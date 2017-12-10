import numpy as np
import mungetools as mg
from sklearn.svm import SVC

def trainClassifier(DF,paramc,paramg,split=0.5):

    nvals = len(DF)
    splitind=np.floor(nvals*split)
    nparams = len(paramc)*len(paramg)
    scores = np.zeros(nparams)
    counter=0
    paramholder=np.zeros([nparams,2])

    rp = np.random.permutation(nvals)
    survt=np.array(DF.iloc[rp[0:splitind],0])
    survcv=np.array(DF.iloc[rp[splitind:nvals],0])

    survt[~survt]=-1
    survcv[~survcv]=-1
    tset = DF.iloc[rp[0:splitind],1:]
    cvset = DF.iloc[rp[splitind:nvals],1:]
    bestscore=-1

    for c in paramc:
        for g in paramg:
            model=SVC(C=c,gamma=g)
            model=model.fit(tset,survt)
            try:
                scorei=model.score(cvset,survcv)
            except:
                scorei=0 
            scores[counter]=np.mean(scorei)
            paramholder[counter,0]=c
            paramholder[counter,1]=g
            if scorei>bestscore:
                bestscore=scorei
                bestmodel=model
            counter+=1
            print('Score = %f with c: %f, g: %f' %(scorei,c,g))
    bestc=paramholder[scores.argmax(),0]
    bestg=paramholder[scores.argmax(),1]
    print('Best score of %f with c: %f, g: %f' %(bestscore,bestc,bestg))
    return bestmodel


trdata,testdata=mg.loadData()

testid = np.array(testdata.PassengerId)

trdata,tesrdata=mg.addFamSurvivors(trdata,testdata)

trdata=mg.mungeData(trdata)
testdata=mg.mungeData(testdata)


testc = [0.05, 0.1, 0.3, 0.6, 1, 3, 5, 10 ]
testg = [0, 0.01, 0.05, 0.1, 0.5, 1, 1.5]

model= trainClassifier(trdata,testc,testg)

preds = model.predict(testdata)
preds=(preds>0)*1 # predictions are -1 and 1, so make 0 and 1
mg.writeout(preds,testid,'predictions/svmmodel_test.csv')
