## Description of the Data and Data Limitations

Our data included two datasets, one with metadata about the semizdad and exile literature (11.614 datapoints) and one with the full text of these publications (10.801 datapoints). The original publications in the 20th century were written by typewriter. These publications were digitalized by optical character regogniztion (ocr).

For the topic modelling, we included only publications between the years 1968 and 1989. In this timespan, there were publication from the samizdat and the exile literature set available, which opened up the opportunity of comparing if the authors writing in and outside of Czech Slovakia might have addressed different topics.

We excluded literature where:
- the literature hat was not written in the timespan or no publication year was available
- where no ocr was available or the ocr left no sensible data, even after some preliminary data cleaning
- where the text file only contained a table of content and no actual continuous text / journal articles
- where the text file contained less than 100 characters (because that signaled to us a lack of content in the document).

All in all, the topic modelling was run on 5.642 datapoints.
Even after some preliminary data cleaning it should be mentioned that a better ocr of the typewritten text might output slightly different output in the topic modelling.


- The original Meta-dataset contains over 11.614 datapoints. 1.468 of these datapoints contained no textual data, as they were either not ocr'ed or not machine readable.

 where as the datasets with the fulltext of the journals only contains around 10.801 datapoints.
- The ocr (optical character recognition) of the journals contains errors that we werent able to correct.
- For the journals there are some (how many?) where the publication year was missing in the meta-data. We were able to add a publication year to some, but not for all of them. We exluded these of our analysis.