def backmatter():
    return [
        ('h1', 'CRediT authorship contribution statement'),
        ('p', "**Rohul Amin:** Conceptualization, Methodology, Software, Formal analysis, Investigation, Data curation, "
              "Visualization, Writing – original draft. **Md. Mehedi Rahman Rana:** Methodology, Validation, Writing – review "
              "& editing, Supervision. **Sumya Akter:** Investigation, Validation, Writing – review & editing."),
        ('h1', 'Declaration of competing interest'),
        ('p', "The authors declare that they have no known competing financial interests or personal relationships that could "
              "have appeared to influence the work reported in this paper."),
        ('h1', 'Funding'),
        ('p', "This research did not receive any specific grant from funding agencies in the public, commercial, or "
              "not-for-profit sectors."),
        ('h1', 'Data availability'),
        ('plain', "The source data are openly available: the MPEA database from the CitrineInformatics/MPEA_dataset GitHub "
              "repository [@mpea_github] and the GE refractory-alloy dataset from figshare [@ge_figshare]. The curated "
              "records, composition features, frozen evaluation splits, per-fold metrics of all models, per-record "
              "predictions, the Kaggle notebooks and the analysis code are provided as Supplementary Material and available "
              "at https://github.com/labidz/hea_mpea_benchmark.", {'align': 'left'}),
        ('h1', 'Declaration of generative AI and AI-assisted technologies in the manuscript preparation process'),
        ('p', "During the preparation of this work the authors used Claude (Anthropic) to assist with writing analysis code, "
              "organizing the computational experiments and drafting and editing the text and figures. After using this "
              "tool, the authors reviewed and edited the content as needed and take full responsibility for the content of "
              "the published article."),
    ]


def acknowledgements():
    return [('h1', 'Acknowledgements'),
            ('p', "The authors thank the compilers of the open MPEA database (Borg et al.) and the GE Research team (Detor et "
                  "al.) for making their data publicly available, and Kaggle for providing GPU computing resources.")]
