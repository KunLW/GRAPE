%% notes.cls — lightweight lecture-notes class (article-derived), dark theme
%% Self-contained styling for technical physics notes.
%% Dark-theme twin of simple-notes.cls: same layout/packages, only the
%% colour layer differs. Keep structural changes in sync with the light class.
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{simple-notes.dark}[2025/06/14 v1.0 lecture notes class (dark)]

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
\definecolor{pagebg}{RGB}{30,32,36}     % page background (soft charcoal)
\definecolor{pagefg}{RGB}{219,221,226}  % default text
\definecolor{linkblue}{RGB}{110,168,254}
\definecolor{rulegray}{RGB}{150,150,150}
\AtBeginDocument{\pagecolor{pagebg}\color{pagefg}}
\tikzset{every picture/.style={color=pagefg}}

% --- Lists, headers, sectioning -----------------------------------
\RequirePackage{enumitem}
\setlist{itemsep=2pt,topsep=3pt}
\RequirePackage{fancyhdr}
\RequirePackage{titlesec}
\titleformat{\section}{\normalfont\large\bfseries}{\thesection}{0.7em}{}
\titleformat{\subsection}{\centering\normalfont\normalsize\bfseries}{\thesubsection}{0.6em}{}
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
\newtcolorbox{keyresult}[1][]{colback=linkblue!12!pagebg,colframe=linkblue!75!pagebg,
  coltext=pagefg,boxrule=0.6pt,arc=2pt,left=6pt,right=6pt,top=4pt,bottom=4pt,#1}
\newtcolorbox{notebox}[1][]{colback=orange!12!pagebg,colframe=orange!70,
  coltext=pagefg,coltitle=pagebg,boxrule=0.6pt,arc=2pt,left=6pt,right=6pt,top=4pt,bottom=4pt,
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
