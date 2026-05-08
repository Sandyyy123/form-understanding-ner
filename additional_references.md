# Additional References - Project 18 (FUNSD Form Understanding)

Independent literature scout output. All entries below were resolved live against the CrossRef API (`https://api.crossref.org/works/{doi}`) at the time of writing. Per project rule, only Author / Title / Journal / Year / DOI are kept; volume / issue / page numbers are intentionally omitted.

The existing `reports/references.md` was inspected ONLY for the SOTA-gap callout below and was NOT modified.

## State-of-the-art callout (gaps the current `references.md` does not cover)

The existing references stop at the LayoutLMv3 / Donut / Pix2Struct era and at the LayoutXLM survey (2021-2023). Five concrete gaps that Project 18 should cite:

1. **DocLLM (Wang et al., ACL 2024)** - the first decoder-only generative LLM that ingests bounding-box layout via disentangled spatial attention, beating LayoutLMv3 on KIE without an image encoder. Directly relevant as a "BERT vs LayoutLMv3 vs DocLLM" three-way comparison would strengthen the manuscript.
2. **LayoutLLM (Luo et al., CVPR 2024 + Fujitsu IJDAR 2024)** - layout instruction-tuning; closes the gap between encoder-only LayoutLMv3 and modern instruction-tuned VLMs on FUNSD-style entity tagging.
3. **mPLUG-DocOwl 1.5 / 2 (Hu et al., EMNLP-Findings 2024 / ACL 2025)** - sets the current OCR-free bar on form / receipt benchmarks; should be discussed as an alternative to the LayoutLMv3 advanced baseline.
4. **3MVRD (Ding et al., ACL-Findings 2024)** - multi-task multi-teacher distillation specifically targeting form documents (FUNSD lineage). Directly applicable methodology.
5. **VRD-IU shared-task lessons (Ding et al., IJCAI 2024)** and **MosaicDoc benchmark (Chen et al., AAAI 2026)** - newer form-understanding benchmarks beyond FUNSD that the manuscript's "future work" should reference.

In addition, the current references list no 2024-2026 IDP survey. Adding **Gbada et al., IJDAR 2024** and **Ding et al., AI Review 2026** would close the survey gap.

---

## Architectures and methods (2024-2026)

1. Wang D, Raman N, Sibue M, Ma Z, Babkin P, Kaur S. DocLLM: A Layout-Aware Generative Language Model for Multimodal Document Understanding. Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2024. DOI:10.18653/v1/2024.acl-long.463
2. Luo C, Shen Y, Zhu Z, Zheng Q, Yu Z, Yao C. LayoutLLM: Layout Instruction Tuning with Large Language Models for Document Understanding. 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 2024. DOI:10.1109/cvpr52733.2024.01480
3. Hu A, Xu H, Ye J, Yan M, Zhang L, Zhang B. mPLUG-DocOwl 1.5: Unified Structure Learning for OCR-free Document Understanding. Findings of the Association for Computational Linguistics: EMNLP 2024. 2024. DOI:10.18653/v1/2024.findings-emnlp.175
4. Hu A, Xu H, Zhang L, Ye J, Yan M, Zhang J. mPLUG-DocOwl2: High-resolution Compressing for OCR-free Multi-page Document Understanding. Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2025. DOI:10.18653/v1/2025.acl-long.291
5. Liu Y, Yang B, Liu Q, Li Z, Ma Z, Zhang S. TextMonkey: An OCR-Free Large Multimodal Model for Understanding Document. IEEE Transactions on Pattern Analysis and Machine Intelligence. 2026. DOI:10.1109/tpami.2026.3653415
6. Ding Y, Vaiani L, Han C, Lee J, Garza P, Poon J. 3MVRD: Multimodal Multi-task Multi-teacher Visually-Rich Form Document Understanding. Findings of the Association for Computational Linguistics ACL 2024. 2024. DOI:10.18653/v1/2024.findings-acl.903
7. Adnan W, Tang J, Zouggari Y, Laatiri S, Lam L, Caspani F. A LayoutLMv3-Based Model for Enhanced Relation Extraction in Visually-Rich Documents. Lecture Notes in Computer Science (ICDAR 2024). 2024. DOI:10.1007/978-3-031-70546-5_10
8. The H, Hoai V, Yang J. One-Shot Transformer-Based Framework for Visually-Rich Document Understanding. Lecture Notes in Computer Science (ICDAR 2024). 2024. DOI:10.1007/978-3-031-70533-5_15
9. Yang J, The H, Tuan H. Light-Weight Multi-modality Feature Fusion Network for Visually-Rich Document Understanding. Lecture Notes in Computer Science (ICDAR 2024). 2024. DOI:10.1007/978-3-031-70533-5_12
10. Van Landeghem J, Maity S, Banerjee A, Blaschko M, Moens M, Lladós J. DistilDoc: Knowledge Distillation for Visually-Rich Document Applications. Lecture Notes in Computer Science (ICDAR 2024). 2024. DOI:10.1007/978-3-031-70546-5_12
11. Luo C, Tang G, Zheng Q, Yao C, Jin L, Li C. Bi-VLDoc: bidirectional vision-language modeling for visually-rich document understanding. International Journal on Document Analysis and Recognition (IJDAR). 2025. DOI:10.1007/s10032-025-00518-w
12. Arshad A, Moetesum M, Hasan A, Shafait F. A Graph-Augmented Multi-Stage Transformer Model for Document Layout Understanding. International Journal on Document Analysis and Recognition (IJDAR). 2025. DOI:10.1007/s10032-025-00566-2
13. Trivedi A, Khanna S, Chaudhury S, Harit G. Representation learning approach for understanding structured documents. Scientific Reports. 2025. DOI:10.1038/s41598-025-33642-y
14. Zhang C, Tu Y, Zhao Y, Yuan C, Chen H, Zhang Y. Modeling Layout Reading Order as Ordering Relations for Visually-rich Document Understanding. Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing. 2024. DOI:10.18653/v1/2024.emnlp-main.540
15. Zhang W, Liu F, Xu Z, Bi Z. Geometry-aware and autonomous reading-order learning for OCR-free document parsing. International Journal of Web Information Systems. 2025. DOI:10.1108/ijwis-07-2025-0204
16. Le B, Xu S, Fu J, Huang Z, Li M, Guo Y. QID: Efficient Query-Informed ViTs in Data-Scarce Regimes for OCR-Free Visual Document Understanding. 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW). 2025. DOI:10.1109/cvprw67362.2025.00014

## Key information extraction and graph-based approaches

17. Majumder R, Wang Z, Yue Y, Kalita M, Liu J. Enforcing Graph Structures to Enhance Key Information Extraction in Document Analysis. Proceedings of the 20th International Joint Conference on Computer Vision, Imaging and Computer Graphics Theory and Applications. 2025. DOI:10.5220/0013240600003912
18. Velampalli S. LLM-assisted Automatic Feature Extraction for Document Understanding and Analytics. Proceedings of the 2025 ACM Symposium on Document Engineering. 2025. DOI:10.1145/3704268.3749108
19. Scius-Bertrand A, Fakhari A, Vögtlin L, Cabral D, Fischer A. Are Layout Analysis and OCR Still Useful for Document Information Extraction Using Foundation Models? Lecture Notes in Computer Science (ICDAR 2024). 2024. DOI:10.1007/978-3-031-70546-5_11
20. Zhang C, Zhao Y, Xie Y, Yuan C, Tu Y, Guo Y. Unveiling the Deficiencies of Pre-trained Text-and-Layout Models in Real-world Visually-rich Document Information Extraction. Findings of the Association for Computational Linguistics: EACL 2026. 2026. DOI:10.18653/v1/2026.findings-eacl.1

## Datasets and benchmarks

21. Ding Y, Han S, Li Y, Poon J. VRD-IU: Lessons from Visually Rich Document Intelligence and Understanding. Proceedings of the Thirty-Third International Joint Conference on Artificial Intelligence (IJCAI). 2024. DOI:10.24963/ijcai.2024/1258
22. Chen K, Chen Y, Xue Y. MosaicDoc: A Large-Scale Bilingual Benchmark for Visually Rich Document Understanding. Proceedings of the AAAI Conference on Artificial Intelligence. 2026. DOI:10.1609/aaai.v40i4.37282
23. Yang Z, Tang J, Li Z, Wang P, Wan J, Zhong H. CC-OCR: A Comprehensive and Challenging OCR Benchmark for Evaluating Large Multimodal Models in Literacy. 2025 IEEE/CVF International Conference on Computer Vision (ICCV). 2025. DOI:10.1109/iccv51701.2025.02019
24. Huybrechts G, Ronanki S, Jayanthi S, Fitzgerald J, Veeravanallur S. Document Haystack: A Long Context Multimodal Image/Document Understanding Vision LLM Benchmark. 2025 IEEE/CVF International Conference on Computer Vision Workshops (ICCVW). 2025. DOI:10.1109/iccvw69036.2025.00428
25. Gunathilaka K, Hewagama D, Pushpakumara S, Ambegoda T. SinFUND and SinOCR: Benchmarks for Sinhala Handwritten OCR and Template-Free Form Understanding. Research Square preprint. 2025. DOI:10.21203/rs.3.rs-6976719/v1
26. Xiao B, Simsek M, Kantarci B, Alkheir A. Revisiting Table Detection Datasets for Visually Rich Documents. International Journal on Document Analysis and Recognition (IJDAR). 2025. DOI:10.1007/s10032-025-00527-9
27. Singh L, Middleton S. Tabular context-aware optical character recognition and tabular data reconstruction for historical records. International Journal on Document Analysis and Recognition (IJDAR). 2025. DOI:10.1007/s10032-025-00543-9

## Surveys (2024-2026)

28. Gbada H, Kalti K, Mahjoub M. Deep learning approaches for information extraction from visually rich documents: datasets, challenges and methods. International Journal on Document Analysis and Recognition (IJDAR). 2024. DOI:10.1007/s10032-024-00493-8
29. Ding Y, Han S, Lee J, Hovy E. Deep learning based visually rich document content understanding: a survey. Artificial Intelligence Review. 2026. DOI:10.1007/s10462-025-11477-3
30. Zhang X. Roles of MLLMs in Visually Rich Document Retrieval for RAG: A Survey. Proceedings of the 14th International Joint Conference on Natural Language Processing (IJCNLP-AACL). 2025. DOI:10.18653/v1/2025.ijcnlp-long.2

## Related table-structure work (relevant to KIE on forms)

31. Xiao B, Simsek M, Kantarci B, Alkheir A. Rethinking detection based table structure recognition for visually rich document images. Expert Systems with Applications. 2025. DOI:10.1016/j.eswa.2025.126461

---

## Summary

- 31 papers, every one resolved live via CrossRef.
- Coverage: 14 architecture/method papers, 4 KIE/graph papers, 7 dataset/benchmark papers, 3 surveys, 1 table-structure paper.
- Year mix: 13 from 2024, 14 from 2025, 4 from 2026.
- Five SOTA gaps identified vs the existing `reports/references.md`: DocLLM, LayoutLLM, mPLUG-DocOwl 1.5/2, 3MVRD, MosaicDoc / VRD-IU benchmarks.
