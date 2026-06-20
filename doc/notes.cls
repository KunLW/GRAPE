% LaTeX class file for physics notes with Chinese support
% Author: Kun
% Date: 2025年9月15日

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{notes}[2025/09/15 Physics Notes Class with Chinese Support]

% Define class options for font scaling
\RequirePackage{kvoptions}
\SetupKeyvalOptions{
    family=notes,
    prefix=notes@
}

% Font scale option - default is 1.0 (normal size)
\DeclareStringOption[1.0]{fontscale}
\ProcessKeyvalOptions*

% Base class
\LoadClass[11pt,b5paper]{article}

% Required packages for Chinese support and fonts
\RequirePackage{silence}
\WarningFilter{xeCJK}{Redefining CJKfamily}
\RequirePackage{fontspec}               % Must be loaded before xeCJK
\RequirePackage[UTF8,heading=true,scheme=chinese]{ctex}
\RequirePackage{xeCJK}

% Mathematics and physics packages
\RequirePackage{amsmath}
\RequirePackage{amsfonts}
\RequirePackage{amssymb}
\RequirePackage{amsthm}
\RequirePackage{mathtools}
\usepackage{physics}
\RequirePackage{tensor}
\RequirePackage{braket}
% Uncomment the following line if you prefer siunitx version of \qty
\renewcommand{\textsc}[1]{{\fontspec{BodoniSvtyTwoSCITCTT-Book}#1}}
% Graphics and figures
\RequirePackage{graphicx}
\RequirePackage{float}
\RequirePackage{subcaption}
\RequirePackage{tikz}
\RequirePackage{pgfplots}
\pgfplotsset{compat=1.18}

% Page layout and formatting
\RequirePackage[
  top=2.2cm, bottom=2.5cm,
  left=2.2cm, right=2.2cm,
  headheight=14.5pt, headsep=8mm, footskip=10mm
]{geometry}
\RequirePackage{fancyhdr}
\RequirePackage{titlesec}
\RequirePackage{enumitem}
\RequirePackage{multicol}

% Colors and styling
\RequirePackage{xcolor}
\RequirePackage{mdframed}
\RequirePackage{tcolorbox}
\tcbuselibrary{theorems,skins,breakable}

% Other useful packages
\RequirePackage{hyperref}
\RequirePackage{cleveref}
\RequirePackage{booktabs}
\RequirePackage{array}
\RequirePackage{listings}

% Color definitions
\definecolor{physicsblue}{RGB}{0,51,102}
\definecolor{physicsgreen}{RGB}{0,102,51}
\definecolor{physicsred}{RGB}{153,0,51}
\definecolor{lightgray}{RGB}{245,245,245}
\definecolor{darkgray}{RGB}{100,100,100}

% Configure hyperref
\hypersetup{
    colorlinks=true,
    linkcolor=physicsblue,
    filecolor=physicsgreen,
    urlcolor=physicsred,
    citecolor=physicsblue,
    pdftitle={Physics Notes},
    pdfauthor={Kun}
}

% Font settings - separate English and Chinese fonts
% Global font size scaling - controlled by fontscale parameter
\RequirePackage{anyfontsize}
\RequirePackage{calc}

% Calculate scaled font sizes based on fontscale parameter
\newlength{\basefontsize}
\setlength{\basefontsize}{11pt * \real{\notes@fontscale}}

\newlength{\smallfontsize}
\setlength{\smallfontsize}{10pt * \real{\notes@fontscale}}

\newlength{\footnotefontsize}
\setlength{\footnotefontsize}{8pt * \real{\notes@fontscale}}

\newlength{\largefontsize}
\setlength{\largefontsize}{14pt * \real{\notes@fontscale}}

\newlength{\Largefontsize}
\setlength{\Largefontsize}{16pt * \real{\notes@fontscale}}

\newlength{\LARGEfontsize}
\setlength{\LARGEfontsize}{18pt * \real{\notes@fontscale}}

\newlength{\hugefontsize}
\setlength{\hugefontsize}{22pt * \real{\notes@fontscale}}

\newlength{\Hugefontsize}
\setlength{\Hugefontsize}{26pt * \real{\notes@fontscale}}

% Apply scaled font sizes
\renewcommand{\normalsize}{\fontsize{\basefontsize}{1.2\basefontsize}\selectfont}
\renewcommand{\small}{\fontsize{\smallfontsize}{1.2\smallfontsize}\selectfont}
\renewcommand{\footnotesize}{\fontsize{\footnotefontsize}{1.2\footnotefontsize}\selectfont}
\renewcommand{\large}{\fontsize{\largefontsize}{1.2\largefontsize}\selectfont}
\renewcommand{\Large}{\fontsize{\Largefontsize}{1.2\Largefontsize}\selectfont}
\renewcommand{\LARGE}{\fontsize{\LARGEfontsize}{1.2\LARGEfontsize}\selectfont}
\renewcommand{\huge}{\fontsize{\hugefontsize}{1.2\hugefontsize}\selectfont}
\renewcommand{\Huge}{\fontsize{\Hugefontsize}{1.2\Hugefontsize}\selectfont}

% English fonts - you can change these to your preferred fonts
\setmainfont{Times New Roman}           % English main font
% Other options: Palatino, Minion Pro, Adobe Garamond Pro, etc.
\setsansfont{Helvetica}                 % English sans serif font  
% Other options: Arial, Calibri, Source Sans Pro, etc.
\setmonofont{Menlo}                     % English monospace font
% Other options: Monaco, Courier New, Source Code Pro, etc.

% Chinese font settings - use system fonts available on macOS
\setCJKmainfont{方正金陵}[
    ItalicFont={行楷-简}  % Use Xingkai for italic text
]
\setCJKsansfont{方正灵飞刻石 简}
\setCJKmonofont{方正灵飞刻石 简}
% Other Chinese font options: 
% 楷体, 宋体, 黑体, 仿宋, 华文楷体, 华文宋体, etc.

% Page style
\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\leftmark}
\fancyhead[R]{\thepage}
\fancyfoot[C]{\footnotesize Kun's Notes - \today}
\renewcommand{\headrulewidth}{0.4pt}
\renewcommand{\footrulewidth}{0.2pt}
\setlength{\headheight}{14.39995pt}  % Fix header height warning

% Section formatting
\titleformat{\section}
{\Large\bfseries\color{physicsblue}}
{\thesection}{1em}{}

\titleformat{\subsection}
{\large\bfseries\color{physicsgreen}}
{\thesubsection}{1em}{}

\titleformat{\subsubsection}
{\normalsize\bfseries\color{physicsred}}
{\thesubsubsection}{1em}{}
\definecolor{lightSky}{RGB}{180,210,255} % 一种特别柔和的天蓝色

% Theorem environments
\theoremstyle{definition}
\newtheorem{definition}{定义}[section]
\newtheorem{theorem}{定理}[section]
\newtheorem{lemma}{引理}[section]
\newtheorem{corollary}{推论}[section]
\newtheorem{proposition}{命题}[section]

\theoremstyle{remark}
\newtheorem{remark}{注解}[section]
\newtheorem{exercise}{练习}[section]

\newmdtheoremenv[
  backgroundcolor=lightSky!20,
  linecolor=lightSky!20,
  roundcorner=5pt,
]{example}{示例}

% Custom tcolorbox environments
\newtcolorbox{importantbox}[1][]{
    colback=physicsblue!5!white,
    colframe=physicsblue!75!black,
    title=重要,
    fonttitle=\bfseries,
    #1
}

\newtcolorbox{formulabox}[1][]{
    colback=physicsgreen!5!white,
    colframe=physicsgreen!75!black,
    title=公式,
    fonttitle=\bfseries,
    #1
}

\newtcolorbox{conceptbox}[1][]{
    colback=physicsred!5!white,
    colframe=physicsred!75!black,
    title=概念,
    fonttitle=\bfseries,
    #1
}

\newtcolorbox{notebox}[1][]{
    colback=orange!5!white,
    colframe=orange!75!black,
    title=笔记,
    fonttitle=\bfseries,
    #1
}

% Custom commands for physics notation
\newcommand{\ve}[1]{\boldsymbol{#1}}  % Vector
\newcommand{\un}[1]{\hat{#1}}         % Unit vector
\newcommand{\pd}[2]{\frac{\partial #1}{\partial #2}}  % Partial derivative
\newcommand{\td}[2]{\frac{d #1}{d #2}}                % Total derivative
\renewcommand{\grad}{\nabla}          % Gradient (redefine to avoid conflict)
\newcommand{\divg}{\nabla \cdot}
\renewcommand{\curl}{\nabla \times}   % Curl (redefine to avoid conflict)
\newcommand{\lapl}{\nabla^2}

% Constants
\newcommand{\kb}{k_\mathrm{B}}    % Boltzmann constant
\newcommand{\kB}{k_\mathrm{B}}    % Boltzmann constant
% \hbar is already defined by physics package
\newcommand{\eps}{\varepsilon}    % Epsilon
\newcommand{\eo}{\varepsilon_0}   % Permittivity of free space
\newcommand{\muo}{\mu_0}          % Permeability of free space

% Mathematics shortcuts
\newcommand{\R}{\mathbb{R}}       % Real numbers
\newcommand{\C}{\mathbb{C}}       % Complex numbers
\newcommand{\N}{\mathbb{N}}       % Natural numbers
\newcommand{\Z}{\mathbb{Z}}       % Integers
\newcommand{\Q}{\mathbb{Q}}       % Rational numbers

% Custom spacing
\setlength{\parindent}{0pt}
\setlength{\parskip}{4pt plus 2pt minus 1pt}

% List formatting
\setlist[itemize]{leftmargin=*,topsep=2pt,itemsep=1pt}
\setlist[enumerate]{leftmargin=*,topsep=2pt,itemsep=1pt}

% Figure and table captions
\RequirePackage[font=normalsize,labelfont=bf]{caption}

% Code listing settings
\lstset{
    basicstyle=\ttfamily\normalsize,
    backgroundcolor=\color{lightgray},
    frame=single,
    numbers=left,
    numberstyle=\small\color{darkgray},
    breaklines=true,
    showstringspaces=false
}

% Title page customization
\newcommand{\coursetitle}[1]{\def\@coursetitle{#1}}
\newcommand{\semester}[1]{\def\@semester{#1}}
\newcommand{\instructor}[1]{\def\@instructor{#1}}

\renewcommand{\maketitle}{
    \begin{titlepage}
        \centering
        \vspace*{2cm}
        
        {\Huge\bfseries\color{physicsblue} \@title \par}
        \vspace{1cm}
        
        \ifdef{\@coursetitle}{
            {\Large\color{physicsgreen} \@coursetitle \par}
            \vspace{0.5cm}
        }{}
        
        \ifdef{\@semester}{
            {\large \@semester \par}
            \vspace{0.5cm}
        }{}
        
        {\large\bfseries \@author \par}
        \vspace{0.5cm}
        
        \ifdef{\@instructor}{
            {\normalsize 授课教师: \@instructor \par}
            \vspace{0.5cm}
        }{}
        
        {\normalsize \@date \par}
        
        \vfill
        
        {\small\color{darkgray} Physics Notes}
    \end{titlepage}
}

% End of class file