# Automatic Multispeaker Voice Cloning Across Languages

### Papers 
| URL | Designation | Title |
| --- | ------------ | ----- |
|[1808.10128](https://arxiv.org/pdf/1808.10128.pdf) | SST4TTS | Semi-Supervised Training for Improving Data Efficiency in End-to-End Speech Synthesis |
|[**1806.04558**](https://arxiv.org/pdf/1806.04558.pdf) | **SV2TTS** | **Transfer Learning from Speaker Verification to Multispeaker Text-To-Speech Synthesis** |
|[1802.06006](https://arxiv.org/pdf/1802.06006.pdf) | / | Neural Voice Cloning with a Few Samples |
|[1712.05884](https://arxiv.org/pdf/1712.05884.pdf) | Tacotron 2 | Natural TTS Synthesis by Conditioning Wavenet on Mel Spectrogram Predictions |
|[1710.10467](https://arxiv.org/pdf/1710.10467.pdf) | GE2E | Generalized End-To-End Loss for Speaker Verification |
|[1710.07654](https://arxiv.org/pdf/1710.07654.pdf) | Deep Voice 3 | Deep Voice 3: Scaling Text-to-Speech with Convolutional Sequence Learning |
|[1705.08947](https://arxiv.org/pdf/1705.08947.pdf) | Deep Voice 2 | Deep Voice 2: Multi-Speaker Neural Text-to-Speech |
|[1703.10135](https://arxiv.org/pdf/1703.10135.pdf) | Tacotron | Tacotron: Towards End-To-End Speech Synthesis |
|[1702.07825](https://arxiv.org/pdf/1702.07825.pdf) | Deep Voice 1 | Deep Voice: Real-time Neural Text-to-Speech |
|[1609.03499](https://arxiv.org/pdf/1609.03499.pdf) | Wavenet | Wavenet: A Generative Model for Raw Audio |
|[1509.08062](https://arxiv.org/pdf/1509.08062.pdf) | TE2E | End-to-End Text-Dependent Speaker Verification |
|[1506.07503](https://arxiv.org/pdf/1506.07503.pdf) | Attention (location) | Attention-Based Models for Speech Recognition |
|[1409.0473](https://arxiv.org/pdf/1409.0473.pdf) | Attention (basic) | Neural Machine Translation by Jointly Learning to Align and Translate |


### Task list
*In no particular order:*
- Other papers to read:
  - [x] Tacotron 2 (base for the synthesizer and vocoder of SV2TTS)
  - [x] GE2E (Encoder of SV2TTS)
  - [x] TE2E (base for GE2E)
  - [x] Attention (basic)
  - [ ] Attention (location)
  - [x] Tacotron 1 (base for Tacotron 2)
  - [x] Wavenet (vocoder of Tacotron)
- SOTA review:
  - [x] HMM-based TTS
  - [x] DNN-based TTS
  - [x] RNN-based TTS
  - [x] Wavenet
  - [x] Deep voice
  - [x] Tacotron2
  - [ ] SV2TTS
  - ... more?
- [ ] Get started on the description of SV2TTS 
- [ ] Get started on the analysis of the benchmarks in SV2TTS
- On the encoder:
  - [ ] Present the implementation and that of the authors
  - [ ] Present the results
- On the Tacotron 2 implementation:
  - [ ] Present results from the basic implementation
  - [ ] Address the Wavenet speed problem
  - [ ] Train on custom data
	

### Roadmap
**For the 25th january**:
- Finish the encoder (GE2E) and write about it
- Present the results

**For the end of february**:
- Integrate the encoder with Tacotron
- Use Griffin-Lim to generate waveforms in a first time

**For later**
- Possibly, manage to use Wavenet as the waveform generator

**For much later**:
- Implement the improvements in SST4TTS to achieve better data efficiency
- Adapt/Improve the cross-language aspect from the baseline

### Other things
- Migrate repo to github once the baseline is decent *(possibly, make it an open source repo on its own and keep working on transfer across languages as a fork)*
