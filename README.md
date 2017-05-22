# Movie Genre Classification from Subtitles

In this project, aim is to categorise movies into genres by analysing subtitles
with machine learning techniques. These techniques focus several aspects of the
subtitles such as sound descriptions, dialogue frequency and raw subtitle scripts.
Some of the applications offer novel solution for movie genre classification problem.

In this project we attempt to classify movies into a set of genres by analysing subtitles of the movies in distinct ways since subtitles are captions of the movies’ transcripts. We approach this problem from three main perspectives. 

> (i) The most straight-forward method is to consider the subtitle texts as collection of words disregarding the grammar and word order. This produces bag of words that contains the multiplicity of the all words occur in input data. Since word order is not taken into account, time related information for the movie is also not considered. We have implemented methods that use this approach and this will probably have the most effect in our final model.

> (ii) The second approach is to use the descriptions that exist for people who has hearing impairment exist in certain subtitles. In order not to run into problems we primarily downloaded subtitles that have this kind of sound descriptions. There are a few amount of subtitles that do not have hearing impaired sound descriptions therefore we simply ignore these subtitles when training. We sift out the sound descriptions from the text and ran the same learning algorithm as the first method. The idea of the second  approach is similar the the first one because both of them ignore the time factor and directly try to learn chunks of words. 

> (iii) The third approach is a novel approach that analyses the speaking frequency as a whole and also in certain periods. This approach allows us to make a connection between time and text which differentiates the project from machine learning applications on plain text.

![Alt text](/../master/figures/figure_logic.png)

### For more information about this project,
Project Report link:
[Project Report](../master/Group4_Final_Report.pdf)

Project Final Presentation link:
[Project Presentation](../master/Group4_Final_Presentation.pptx)
