# Codes of USENIX Security'23 paper "On the Security Risks of Knowledge Graph Reasoning"

(TO COMPLETE)

## Sources

**Constructing data**

- Cyber KG

We construct Cyber KG with recorded CVEs ([link](https://www.cvedetails.com/browse-by-date.php)), from which we crawled vulnerability-related information such as affected vendor, product, version, vulnerability types, descriptions, relevant CWE, etc. One can refer to `./data/cyberkg/crawler.ipynb` to check the information we crawled. We construct a Cybersecurity KG with queries/answers in `gen_cyberkg.py`.

We modify on the released codes from [this repository](https://github.com/snap-stanford/KGReasoning).

## Guide

We organize the structure of our files as follows:
```latex
.
├──  data/
│   └──  cyberkg/              # constructed KG may also saved in this dir by default
│       ├──  cve_url/          # collected CVE web links from 1999 to 2019
│       └──  crawler.ipynb     # crawling scripts that takes cve_url/ as inputs
├──  genkg/
│   ├──  cyberkg_backbone.py   # parse crawled data and construct a cyberkg
│   ├──  cyberkg_query.py      # generate queries and answers
│   └──  cyberkg_utils.py      # utility functions specific to generate cyberkg and QA
├──  dataloader.py              
├──  gen_cyberkg.py            # run this file to generate a cyberkg and QA, see details below
├──  main.py                   # main file for reasoning
├──  models.py                  
├──  READEME.md
└──  util.py                    

```

**Run the Code**

To run the code, one needs to first construct a CyberKG and its QA, then feed them to a model for downstream reasoning task.

- **Step1** : You may need the crawled CVE files from [link](https://github.com/HarrialX/knowledge-base), where you need to first download the `./data/cyberkg-raw/` into you local disk recorded as `<raw path>`.

- **Step2** : To construct a CyberKG with corresponding QA, one can directly run `python gen_cyberkg.py --raw_path <raw path>`, where the `<raw path>` are the load path of your saved raw CVE information. Instead of using all crawled CVE-IDs, our codes filter them with specific vendors and products: if a vendor has number of products within a threshold, and when a product has number of versions within another threshold, we will keep them and keep their related CVE-IDs. We then remove the graph edges (konwledge facts) that contains removed entities. You can adjust those two thresholds by `python gen_cyberkg.py --raw_path <raw path> --pd_num_thre <low int boundary> <high int boundary> --ver_num_thre <low int boundary> <high int boundary>`.

- **Step3** : After generating train/test queries and answers, you can use the exampled commands presented in the [original repository](https://github.com/snap-stanford/KGReasoning).

We also provide a runnable demo in `demo.sh` for an easily use, but you have to download the crawled CVE files from [link](https://github.com/HarrialX/knowledge-base) and change argparser `raw_path` in `gen_cyberkg.py`.


## Cite
Please cite our work:
> @inproceedings{kg-attack,
  title="{On the Security Risks of Knowledge Graph Reasoning}",
  author={Xi, Zhaohan and Du, Tianyu and Pang, Ren and Li, Changjiang and Ji, Shouling and Luo, Xiapu and Xiao, Xusheng and Ma, Fenglong and Wang, Ting},
  booktitle={Proceedings of USENIX Security Symposium (SEC)},
  year={2023}
}

