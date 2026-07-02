%% notes.cls — lightweight lecture-notes class (article-derived)
%% Self-contained styling for technical physics notes.
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{notes2}[2025/06/14 v1.0 lecture notes class]

% Pass all options to article (default 11pt).
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}}
\ExecuteOptions{11pt}
\ProcessOptions\relax
\LoadClass{article}

% --- Page geometry ------------------------------------------------
\RequirePackage[a4paper,margin=1in,headheight=14pt]{geometry}

% --- Fonts --------------------------------------------------------
\RequirePackage[T1]{fontenc}
\RequirePackage{mathpazo}      % Palatino text + matching math

% \RequirePackage{newtxtext}
% \RequirePackage{newtxmath}

\RequirePackage{microtype}


% --- Math ---------------------------------------------------------
\RequirePackage{amsmath}
\RequirePackage{amssymb}
\RequirePackage{amsthm}
\RequirePackage{mathtools}
\RequirePackage{bm}

% --- Physics ------------------------------------------------------
\RequirePackage{physics}

% --- Graphics / colour --------------------------------------------
\RequirePackage{graphicx}
\RequirePackage{tikz}
\RequirePackage{xcolor}
\definecolor{linkblue}{RGB}{20,80,160}
\definecolor{rulegray}{RGB}{120,120,120}

% --- Lists, headers, sectioning -----------------------------------
\RequirePackage{enumitem}
\setlist{itemsep=2pt,topsep=3pt}
\RequirePackage{fancyhdr}
\RequirePackage{titlesec}
\titleformat{\section}{\normalfont\large\bfseries}{\thesection}{0.7em}{}
\titleformat{\subsection}{\normalfont\normalsize\bfseries}{\thesubsection}{0.6em}{}
\titlespacing*{\section}{0pt}{1.4ex plus 1ex minus .2ex}{0.9ex plus .2ex}

\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\small\itshape\nouppercase{\leftmark}}
\fancyhead[R]{\small\thepage}
\renewcommand{\headrulewidth}{0.4pt}
\renewcommand{\footrulewidth}{0pt}
\fancypagestyle{plain}{\fancyhf{}\fancyhead[R]{\small\thepage}%
  \renewcommand{\headrulewidth}{0pt}}

% --- Hyperref (load late) -----------------------------------------
\RequirePackage[colorlinks=true,linkcolor=linkblue,citecolor=linkblue,%
                urlcolor=linkblue,breaklinks=true]{hyperref}

% --- Theorem-like environments ------------------------------------
\theoremstyle{plain}
\newtheorem{theorem}{Theorem}[section]
\newtheorem{lemma}[theorem]{Lemma}
\newtheorem{proposition}[theorem]{Proposition}
\theoremstyle{definition}
\newtheorem{definition}[theorem]{Definition}
\newtheorem{example}[theorem]{Example}
\theoremstyle{remark}
\newtheorem{remark}[theorem]{Remark}

% --- Boxed note helpers -------------------------------------------
\RequirePackage{tcolorbox}
\tcbuselibrary{skins,breakable}
\newtcolorbox{keyresult}[1][]{colback=linkblue!4,colframe=linkblue!55,
  boxrule=0.6pt,arc=2pt,left=6pt,right=6pt,top=4pt,bottom=4pt,#1}
\newtcolorbox{notebox}[1][]{colback=orange!5!white,colframe=orange!70!black,
  boxrule=0.6pt,arc=2pt,left=6pt,right=6pt,top=4pt,bottom=4pt,
  breakable,title=Note,#1}

% --- Title block ---------------------------------------------------
\renewcommand{\maketitle}{%
  \begin{center}
    {\LARGE\bfseries \@title \par}\vspace{0.8em}
    {\large \@author \par}\vspace{0.4em}
    {\normalsize \@date \par}
  \end{center}
  \vspace{0.6em}
  {\color{rulegray}\hrule height 0.6pt}\vspace{1.2em}
}

\endinput
