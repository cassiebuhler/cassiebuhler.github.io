---
title: Software
permalink: /software/
author_profile: true
redirect_from:
  - /software
noindex: true
---

<style type='text/css'>
  h3 {
    font-weight: normal;
    color: #314E52 !important;
  }
  .container {
    position: relative;
    margin-left: 2em;
    margin-top: 15px;
  }
  .vertical {
    border-left: 5px solid #609162;
    height: 100%;
    position: absolute;
    margin-left: -1em;
    margin-top: -1px;
    display: inline-block;
  }
  .partner {
    font-style: italic;
    color: #314E52;
  }
  .tag {
    display: inline-block;
    font-size: 12px;
    padding: 1px 9px;
    border-radius: 10px;
    margin-left: 8px;
    vertical-align: middle;
    border: 1px solid #609162;
    color: #609162;
  }
  .tag.lib {
    background: #609162;
    color: #fff;
  }
  .nested {
    margin-left: 2.2em;
  }
  .nested .vertical {
    border-left: 3px solid #a9c4aa;
  }
  .group-label {
    font-weight: bold;
    color: #314E52;
    margin: 1.5em 0 0 2em;
  }
  br {
    line-height: 200%;
  }
</style>

Open-source software for conservation decision-making and optimization.

### GLEN (Geospatial LLM-Enabled Navigator) <span class="tag lib">Library</span>
<div class="container">
  <div class="vertical"></div>
  <p>
    An open-source geospatial decision-support platform using open-weight language models over cloud-native geospatial data. Users ask questions in plain language and get back maps, tables, and charts, along with the queries and data sources behind them. GLEN runs on modest hardware rather than in commercial data centers, keeping cost, energy use, and privacy concerns low.
  </p>
  <p style="line-height: 85%;">
    <a href="https://schmidtdse.github.io/glen-website/" target="_blank" rel="noopener noreferrer" class="btn--research">Website</a>
    <a href="https://github.com/SchmidtDSE/glen" target="_blank" rel="noopener noreferrer" class="btn--research">Github</a>
    <a href="https://doi.org/10.5281/zenodo.20673849" target="_blank" rel="noopener noreferrer">
      <img src="https://zenodo.org/badge/DOI/10.5281/zenodo.20673849.svg" alt="DOI">
    </a>
  </p>
  <p>
    Boettiger, C., & Buhler, C. K. (2026). <i>GLEN (Geospatial LLM-Enabled Navigator)</i>. Zenodo. https://doi.org/10.5281/zenodo.20673849
  </p>
</div>
<br>

<p class="group-label">Applications using GLEN</p>
<div class="nested" markdown="1">

<p>Interactive decision-support tools co-developed with partner organizations. Each pairs interactive maps with a natural-language agent that runs reproducible SQL over H3-indexed, cloud-native data. Earlier versions predate GLEN and have since been migrated to it.</p>

### California 30x30 Explorer
<div class="container">
  <div class="vertical"></div>
  <p class="partner">Partner: California Biodiversity Network</p>
  <p>
    Supports California's 30x30 Biodiversity Assessment. Explore the state's 30x30 conservation areas, other conserved lands, and non-conserved lands alongside ecological, socio-environmental, and climate data, revealing what is protected, what isn't, and where the gaps are.
  </p>
  <p style="line-height: 85%;">
    <a href="https://ca-30x30.nrp-nautilus.io" target="_blank" rel="noopener noreferrer" class="btn--research">App</a>
    <a href="https://github.com/boettiger-lab/ca-30x30" target="_blank" rel="noopener noreferrer" class="btn--research">Github</a>
    <a href="https://doi.org/10.5281/zenodo.21284132" target="_blank" rel="noopener noreferrer">
      <img src="https://zenodo.org/badge/DOI/10.5281/zenodo.21284132.svg" alt="DOI">
    </a>
  </p>
  <p>
    Buhler, C. K., & Boettiger, C. (2026). <i>California 30x30 Explorer</i>. Zenodo. https://doi.org/10.5281/zenodo.21284132
  </p>
</div>
<br>

### Trust for Public Land Explorer
<div class="container">
  <div class="vertical"></div>
  <p class="partner">Partner: Trust for Public Land · NASA-funded</p>
  <p>
    A national tool joining conservation investment and land-protection ballot measures with biodiversity, carbon, legislative districts, and social vulnerability data. Draws on TPL's Conservation Almanac and LandVote, the U.S. Climate and Economic Justice Screening Tool (Justice40), and the CDC Social Vulnerability Index.
  </p>
  <p style="line-height: 85%;">
    <a href="https://tpl.nrp-nautilus.io" target="_blank" rel="noopener noreferrer" class="btn--research">App</a>
    <a href="https://github.com/boettiger-lab/tpl" target="_blank" rel="noopener noreferrer" class="btn--research">Github</a>
    <a href="https://doi.org/10.5281/zenodo.18500622" target="_blank" rel="noopener noreferrer">
      <img src="https://zenodo.org/badge/DOI/10.5281/zenodo.18500622.svg" alt="DOI">
    </a>
  </p>
  <p>
    Buhler, C. K., & Boettiger, C. (2026). <i>Trust for Public Land Explorer</i>. Zenodo. https://doi.org/10.5281/zenodo.18500622
  </p>
</div>
<br>

### Trust for Public Land California Explorer
<div class="container">
  <div class="vertical"></div>
  <p class="partner">Partner: Trust for Public Land, California Chapter · NASA-funded</p>
  <p>
    Built for TPL California staff, legislators, and advocates. Explore land conservation investment from the Conservation Almanac, carbon stocks, legislative and congressional districts, and Indigenous lands to support conservation planning and policy advocacy.
  </p>
  <p style="line-height: 85%;">
    <a href="https://tpl-ca.nrp-nautilus.io" target="_blank" rel="noopener noreferrer" class="btn--research">App</a>
    <a href="https://github.com/boettiger-lab/tpl-ca" target="_blank" rel="noopener noreferrer" class="btn--research">Github</a>
    <a href="https://doi.org/10.5281/zenodo.20674186" target="_blank" rel="noopener noreferrer">
      <img src="https://zenodo.org/badge/DOI/10.5281/zenodo.20674186.svg" alt="DOI">
    </a>
  </p>
  <p>
    Boettiger, C., & Buhler, C. K. (2026). <i>Trust for Public Land California Explorer</i>. Zenodo. https://doi.org/10.5281/zenodo.20674186
  </p>
</div>
<br>

### High Seas Explorer
<div class="container">
  <div class="vertical"></div>
  <p class="partner">Partner: Benioff Ocean Science Laboratory</p>
  <p>
    Supports marine protected area planning under the UN High Seas Treaty, in waters beyond national jurisdiction. Explore candidate high seas marine protected areas, exclusive economic zone boundaries, seafloor geomorphology, bathymetry, protected-area coverage, and global fishing effort.
  </p>
  <p style="line-height: 85%;">
    <a href="https://high-seas.nrp-nautilus.io" target="_blank" rel="noopener noreferrer" class="btn--research">App</a>
    <a href="https://github.com/boettiger-lab/bosl-high-seas" target="_blank" rel="noopener noreferrer" class="btn--research">Github</a>
    <a href="https://doi.org/10.5281/zenodo.20674176" target="_blank" rel="noopener noreferrer">
      <img src="https://zenodo.org/badge/DOI/10.5281/zenodo.20674176.svg" alt="DOI">
    </a>
  </p>
  <p>
    Boettiger, C., & Buhler, C. K. (2026). <i>High Seas Explorer</i>. Zenodo. https://doi.org/10.5281/zenodo.20674176
  </p>
</div>
<br>

### Public GYE Explorer
<div class="container">
  <div class="vertical"></div>
  <p class="partner">Greater Yellowstone Ecosystem</p>
  <p>
    Wildlife habitat and public lands across the Greater Yellowstone Ecosystem, centered on Wyoming. Explore Wyoming Game & Fish Department seasonal wildlife ranges, migration corridors, public land management boundaries, and related habitat data.
  </p>
  <p style="line-height: 85%;">
    <a href="https://wyoming-public-demo.nrp-nautilus.io" target="_blank" rel="noopener noreferrer" class="btn--research">App</a>
    <a href="https://github.com/boettiger-lab/wyoming-public-demo" target="_blank" rel="noopener noreferrer" class="btn--research">Github</a>
    <a href="https://doi.org/10.5281/zenodo.20674178" target="_blank" rel="noopener noreferrer">
      <img src="https://zenodo.org/badge/DOI/10.5281/zenodo.20674178.svg" alt="DOI">
    </a>
  </p>
  <p>
    Boettiger, C., & Buhler, C. K. (2026). <i>Public GYE Explorer</i>. Zenodo. https://doi.org/10.5281/zenodo.20674178
  </p>
</div>
<br>

### Earlier versions
<div class="container">
  <div class="vertical"></div>
  <p>
    <b>CA 30x30 Planning & Assessment Tool (California Biodiversity Network Edition)</b>, 2025. Predecessor to the California 30x30 Explorer.<br>
    <a href="https://huggingface.co/spaces/boettiger-lab/ca-30x30-cbn" target="_blank" rel="noopener noreferrer">Hugging Face</a> ·
    <a href="https://github.com/boettiger-lab/cbn-taskforce" target="_blank" rel="noopener noreferrer">Github</a> ·
    <a href="https://doi.org/10.5281/zenodo.16988638" target="_blank" rel="noopener noreferrer">DOI</a>
  </p>
  <p>
    <b>CA 30x30 Planning & Assessment Prototype</b>, 2025. Proof of concept co-designed with California Biodiversity Network participants.<br>
    <a href="https://huggingface.co/spaces/boettiger-lab/ca-30x30" target="_blank" rel="noopener noreferrer">Hugging Face</a> ·
    <a href="https://doi.org/10.5281/zenodo.14933817" target="_blank" rel="noopener noreferrer">DOI</a>
  </p>
  <p>
    <b>LandVote LLM Decision-Support Prototype</b>, 2026. Natural-language exploration of U.S. land conservation ballot measures; predecessor to the Trust for Public Land Explorer.<br>
    <a href="https://huggingface.co/spaces/boettiger-lab/landvote" target="_blank" rel="noopener noreferrer">Hugging Face</a> ·
    <a href="https://github.com/boettiger-lab/landvote" target="_blank" rel="noopener noreferrer">Github</a> ·
    <a href="https://doi.org/10.5281/zenodo.18500783" target="_blank" rel="noopener noreferrer">DOI</a>
  </p>
</div>
<br>

</div>

### Agentic Data Workflows: Cloud-native, AI-ready data preparation on Kubernetes <span class="tag">Data pipeline</span>
<div class="container">
  <div class="vertical"></div>
  <p>
    The data pipeline that prepares the cloud-native, H3-indexed datasets GLEN queries.
  </p>
  <p style="line-height: 85%;">
    <a href="https://github.com/boettiger-lab/data-workflows" target="_blank" rel="noopener noreferrer" class="btn--research">Github</a>
    <a href="https://doi.org/10.5281/zenodo.21466701" target="_blank" rel="noopener noreferrer">
      <img src="https://zenodo.org/badge/DOI/10.5281/zenodo.21466701.svg" alt="DOI">
    </a>
  </p>
  <p>
    Boettiger, C., & Buhler, C. K. (2026). <i>Agentic Data Workflows: Cloud-native, AI-ready data preparation on Kubernetes</i>. Zenodo. https://doi.org/10.5281/zenodo.21466701
  </p>
</div>
<br>

### hybrid-cubic-cg: Nonlinear conjugate gradient method with hybrid cubic regularization <span class="tag">Python package</span>
<div class="container">
  <div class="vertical"></div>
  <p>
    Python package implementing the method in Buhler et al. (2024), <i>Mathematical Programming Computation</i>. A port of Conmin-CG.
  </p>
  <p style="line-height: 85%;">
    <a href="https://pypi.org/project/hybrid-cubic-cg" target="_blank" rel="noopener noreferrer" class="btn--research">PyPI</a>
    <a href="https://doi.org/10.5281/zenodo.23990317" target="_blank" rel="noopener noreferrer">
      <img src="https://zenodo.org/badge/DOI/10.5281/zenodo.23990317.svg" alt="DOI">
    </a>
  </p>
  <p>
    Buhler, C. K., Benson, H. Y., & Shanno, D. F. (2026). <i>hybrid-cubic-cg</i>. Zenodo. https://doi.org/10.5281/zenodo.23990317
  </p>
</div>
<br>

### Conmin-CG: Nonlinear conjugate gradient solver in C <span class="tag">C solver</span>
<div class="container">
  <div class="vertical"></div>
  <p>
    C implementation of the method in Buhler et al. (2024), <i>Mathematical Programming Computation</i>. Selectively applying cubic regularization improves the step quality of conjugate gradient methods; in numerical testing, it reduced iteration counts and runtime compared with the unregularized method.
  </p>
  <p style="line-height: 85%;">
    <a href="https://github.com/cassiebuhler/ConminCG" target="_blank" rel="noopener noreferrer" class="btn--research">Github</a>
    <a href="https://doi.org/10.5281/zenodo.13315591" target="_blank" rel="noopener noreferrer">
      <img src="https://zenodo.org/badge/DOI/10.5281/zenodo.13315591.svg" alt="DOI">
    </a>
  </p>
  <p>
    Buhler, C. K., & Benson, H. Y. (2024). <i>Conmin-CG</i>. Zenodo. https://doi.org/10.5281/zenodo.13315591
  </p>
</div>
<br>

### conservation-dfo: Derivative-free optimization for land conservation <span class="tag">Research code</span>
<div class="container">
  <div class="vertical"></div>
  <p>
    A spatial conservation planning framework using mixed-integer nonlinear programming (MINLP), so it can be paired with ecological software such as population viability analysis. Existing spatial tools typically allow only linear inputs and discrete variables; this framework supports linear <i>and</i> nonlinear inputs, and continuous <i>and</i> discrete variables. Accompanies Buhler and Benson (2024), <i>Proceedings of the AAAI Conference on Artificial Intelligence</i>.
  </p>
  <p style="line-height: 85%;">
    <a href="https://github.com/cassiebuhler/conservation-dfo" target="_blank" rel="noopener noreferrer" class="btn--research">Github</a>
    <a href="https://doi.org/10.5281/zenodo.13742960" target="_blank" rel="noopener noreferrer">
      <img src="https://zenodo.org/badge/DOI/10.5281/zenodo.13742960.svg" alt="DOI">
    </a>
  </p>
  <p>
    Buhler, C. K., & Benson, H. Y. (2024). <i>conservation-dfo</i>. Zenodo. https://doi.org/10.5281/zenodo.13742960
  </p>
</div>
<br>
