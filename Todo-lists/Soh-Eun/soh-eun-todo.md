#### Soh-Eun's To-Do List

Given the variety of languages which are not based on strong prefix and suffixation, it would be desirable to find an alternative that does not rely on the alignment of the infix. My attempt will therefore reframe the multiclass classification problem by substituting the infix with the longest common subsequence (LCS). This draws heavily on prior work by Prof. Mans Hulden, although this work has not, to my knowledge, been applied to morphological inflection as multiclass classification. An example of Prof. Hulden's procedure for extracting the LCS is shown below:

![](../public/LCS.png)

The immediate utility of this approach can be seen in languages such as Arabic, where two example inflection tables are shown as below:

katabtu perf-1-sg
katabta perf-2-m-sg
kutibu pass-perf-3-m-pl
kutibna pass-perf-3-f-pl

darastu perf-1-sg
darasta perf-2-m-sg
durisu pass-perf-3-m-pl
durisna pass-perf-3-f-pl

The morphological paradigms for the examples extend across the entire word with gaps in between, where, for instance, the perfective first-person singular would be x1+a+x2+a+x3+tu. This approach thus constitutes a potential improvement upon the prefix-suffix rule approach in being generalizable to such discontinuous paradigms, where in the case of Arabic, the results of the prefix-suffix rule yielded an accuracy of around 0.7.

My todo list is therefore as follows:

1. Extract paradigm class information with LCS in the SIGMORPHON 2021 data,
2. Follow the prefix-suffix rule approach in framing it as a multiclass classification task, except this time using the aforementioned paradigm information.
