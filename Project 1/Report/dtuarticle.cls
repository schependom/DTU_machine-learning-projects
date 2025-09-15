
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{dtuarticle}[Article class for DTU]
\providecommand{\Logo}{dtu-text}
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}}
\ExecuteOptions{dtu}
\ProcessOptions\relax
\LoadClass{article}
\RequirePackage[a4paper]{geometry}

%\RequirePackage{arev}
%\renewcommand*\familydefault{\sfdefault} %% Only if the base font of the document is to be sans serif
\RequirePackage[T1]{fontenc}
\RequirePackage{hyperref}
\RequirePackage{titling}
\RequirePackage{multicol}
\RequirePackage{graphicx,xcolor}
	\definecolor{grijs}{rgb}{0.25,0.25,0.25}
	\definecolor{dtured}{RGB}{153,0,0}
\RequirePackage{lastpage}
\RequirePackage{tikz}
	\usetikzlibrary{arrows}
	\usetikzlibrary{decorations.pathmorphing}
	\usetikzlibrary{backgrounds}
	\usetikzlibrary{fit}
	\usetikzlibrary{shadows}
	\usetikzlibrary{positioning}
	\usetikzlibrary{calc}
\makeatletter
\newcommand\subtitle[1]{\renewcommand\@subtitle{#1}}
\newcommand\@subtitle{}
\newcommand{\thesubtitle}{\@subtitle{}}
\newcommand\address[1]{\renewcommand\@address{#1}}
\newcommand\@address{}
\newcommand{\theaddress}{\@address{}}
\newcommand\location[1]{\renewcommand\@location{#1}}
\newcommand\@location{}
\newcommand{\thelocation}{\@location{}}
\makeatother
\setlength{\topmargin}{-1in}
\addtolength{\topmargin}{1.5cm}
\setlength{\oddsidemargin}{-1in}
\addtolength{\oddsidemargin}{3cm}
\setlength{\textwidth}{16cm}
\setlength{\textheight}{23.5cm}
\RequirePackage{fancyhdr}
\pagestyle{fancy}
\renewcommand{\headrulewidth}{0pt}
\renewcommand{\footrulewidth}{0pt}
\setlength{\headheight}{1.5cm}
\lhead{}
\chead{}
\rhead{\begin{tikzpicture}[inner sep=0mm]
         \node[anchor=west] at ( 0.5,0.6) {\includegraphics[height=1.5cm]{dtu-logo.pdf}};
      \end{tikzpicture}}
\lfoot{\begin{tabular}[b]{@{}l@{}}
    \footnotesize{\thetitle}\\
    \footnotesize{\theauthor}
  \end{tabular}}
\cfoot{}
\rfoot{\footnotesize{\thepage/\pageref{LastPage}}}

\renewcommand{\maketitle}{
\begin{tikzpicture}[remember picture,overlay]
     \node at (current page.south west)
     {
     \begin{tikzpicture}[remember picture, overlay,inner sep=0mm]
         \fill[black] (17.55,26.87) rectangle (17.65,28.07);
         \node[anchor=north west] at ( 3.0,28.2) {\includegraphics[height=1.5cm]{\Logo}};
         \node[anchor=north east] at ( 19,28.2) {\includegraphics[height=1.5cm]{dtu-logo.pdf}};
         \node[anchor=east] at (17,27.47) {
         \footnotesize\begin{tabular}{@{}r@{}}\theaddress\end{tabular}
         };
         \node[anchor=south east] at (19,24) {
                   \begin{tabular}{@{}r@{}}
                        \large\bf\thetitle\\\textit{\thesubtitle}\\[.5cm]
                        \large\theauthor\vphantom{Ky}
                   \end{tabular}};
         \node[anchor=south west] at (3,24) {
					 \begin{tabular}{@{}l@{}}
									\large \thedate\vphantom{Ky}
							   \end{tabular}};
         \draw  (3,23.7) -- ( 19,23.7);
         \node[opacity=0.5,inner sep=0pt] at ($(current page.south west) + (6cm, 3cm)$){\includegraphics[width=\paperwidth]{math-pattern}};
      \end{tikzpicture}
      };
\end{tikzpicture}
\vspace*{1.5cm}\\
{}
\thispagestyle{empty}
}