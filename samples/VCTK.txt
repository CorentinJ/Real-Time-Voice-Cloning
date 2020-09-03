---------------------------------------------------------------------
                          CSTR VCTK Corpus 
      English Multi-speaker Corpus for CSTR Voice Cloning Toolkit 

                           (Version 0.92) 
                        RELEASE September 2019
             The Centre for Speech Technology Research
                      University of Edinburgh 
                        Copyright (c) 2019 

                         Junichi Yamagishi
                       jyamagis@inf.ed.ac.uk
---------------------------------------------------------------------

Overview 

This CSTR VCTK Corpus includes speech data uttered by 110 English
speakers with various accents. Each speaker reads out about 400
sentences, which were selected from a newspaper, the rainbow passage
and an elicitation paragraph used for the speech accent archive.

The newspaper texts were taken from Herald Glasgow, with permission
from Herald & Times Group. Each speaker has a different set of the
newspaper texts selected based a greedy algorithm that increases the
contextual and phonetic coverage. The details of the text selection 
algorithms are described in the following paper: 

C. Veaux, J. Yamagishi and S. King, 
"The voice bank corpus: Design, collection and data analysis of 
a large regional accent speech database," 
https://doi.org/10.1109/ICSDA.2013.6709856

The rainbow passage and elicitation paragraph are the same for all
speakers. The rainbow passage can be found at International Dialects
of English Archive:
(http://web.ku.edu/~idea/readings/rainbow.htm). The elicitation
paragraph is identical to the one used for the speech accent archive
(http://accent.gmu.edu). The details of the the speech accent archive
can be found at
http://www.ualberta.ca/~aacl2009/PDFs/WeinbergerKunath2009AACL.pdf

All speech data was recorded using an identical recording setup: an
omni-directional microphone (DPA 4035) and a small diaphragm condenser 
microphone with very wide bandwidth (Sennheiser MKH 800), 96kHz 
sampling frequency at 24 bits and in a hemi-anechoic chamber of 
the University of Edinburgh. (However, two speakers, p280 and p315 
had technical issues of the audio recordings using MKH 800). 
All recordings were converted into 16 bits, were downsampled to 
48 kHz, and were manually end-pointed.

This corpus was originally aimed for HMM-based text-to-speech synthesis 
systems, especially for speaker-adaptive HMM-based speech synthesis 
that uses average voice models trained on multiple speakers and speaker
adaptation technologies. This corpus is also suitable for DNN-based 
multi-speaker text-to-speech synthesis systems and waveform modeling.

COPYING 

This corpus is licensed under the Creative Commons License: Attribution 4.0 International 
http://creativecommons.org/licenses/by/4.0/legalcode 

VCTK VARIANTS 
There are several variants of the VCTK corpus: 
Speech enhancement 
- Noisy speech database for training speech enhancement algorithms and TTS models where we added various types of noises to VCTK artificially: http://dx.doi.org/10.7488/ds/2117
- Reverberant speech database for training speech dereverberation algorithms and TTS models where we added various types of reverberantion to VCTK artificially http://dx.doi.org/10.7488/ds/1425
- Noisy reverberant speech database for training speech enhancement algorithms and TTS models http://dx.doi.org/10.7488/ds/2139
- Device Recorded VCTK where speech signals of the VCTK corpus were played back and re-recorded in office environments using relatively inexpensive consumer devices http://dx.doi.org/10.7488/ds/2316
- The Microsoft Scalable Noisy Speech Dataset (MS-SNSD) https://github.com/microsoft/MS-SNSD

ASV and anti-spoofing 
- Spoofing and Anti-Spoofing (SAS) corpus, which is a collection of synthetic speech signals produced by nine techniques, two of which are speech synthesis, and seven are voice conversion. All of them were built using the VCTK corpus. http://dx.doi.org/10.7488/ds/252
- Automatic Speaker Verification Spoofing and Countermeasures Challenge (ASVspoof 2015) Database. This database consists of synthetic speech signals produced by ten techniques and this has been used in the first Automatic Speaker Verification Spoofing and Countermeasures Challenge (ASVspoof 2015) http://dx.doi.org/10.7488/ds/298
- ASVspoof 2019: The 3rd Automatic Speaker Verification Spoofing and Countermeasures Challenge database. This database has been used in the 3rd Automatic Speaker Verification Spoofing and Countermeasures Challenge (ASVspoof 2019) https://doi.org/10.7488/ds/2555


ACKNOWLEDGEMENTS

The CSTR VCTK Corpus was constructed by:
         
        Christophe Veaux   (University of Edinburgh)
        Junichi Yamagishi  (University of Edinburgh)
        Kirsten MacDonald 

The research leading to these results was partly funded from EPSRC
grants EP/I031022/1 (NST) and EP/J002526/1 (CAF), from the RSE-NSFC
grant (61111130120), and from the JST CREST (uDialogue).

Please cite this corpus as follows:
Christophe Veaux,  Junichi Yamagishi, Kirsten MacDonald, 
"CSTR VCTK Corpus: English Multi-speaker Corpus for CSTR Voice Cloning Toolkit",  
The Centre for Speech Technology Research (CSTR), 
University of Edinburgh 

