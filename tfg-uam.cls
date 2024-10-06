\ProvidesClass{tfg-uam}[2017/06/10 v1 LaTeX document class]
%
%
\newif\iflisttables\listtablesfalse
\newif\iflistfigures\listfiguresfalse

\DeclareOption{spanish}{\PassOptionsToPackage{english,main=spanish}{babel}}
\DeclareOption{english}{\PassOptionsToPackage{spanish,main=english}{babel}}

\DeclareOption{listoftables}{\listtablestrue}
\DeclareOption{listoffigures}{\listfigurestrue}

\DeclareOption*{\PassOptionsToClass{\CurrentOption}{book}}
\ProcessOptions

\LoadClass{book}

\RequirePackage[T1]{fontenc}
\RequirePackage{babel}

\RequirePackage[a4paper,left=3.4cm,right=3.4cm,top=3.8cm,bottom=3.8cm]{geometry}

\RequirePackage{url}
\RequirePackage{xcolor}

\definecolor{grisfosc}{gray}{0.25}

\clubpenalty=10000
\widowpenalty=10000


\newcommand{\tutor}[1]{\newcommand{\t@tor}{#1}}
\newcommand{\curso}[1]{\newcommand{\c@rs}{#1}}

\newcommand{\titulacioname}{Grado en Matem\'aticas}
\newcommand{\tfgname}{Trabajo de fin de grado}

\newcommand{\authorname}{Autor}
\newcommand{\tutorname}{Tutor}
\newcommand{\coursename}{Curso}
\newcommand{\keywordsname}{Palabras clave}


\newcommand{\appendicesname}{Apéndices}


\addto\captionsspanish{%
     \renewcommand{\titulacioname}{Grado en Matem\'aticas}
     \renewcommand{\tfgname}{Trabajo de fin de grado}
     \renewcommand{\authorname}{Autor}
     \renewcommand{\tutorname}{Tutor}
     \renewcommand{\coursename}{Curso}
     \renewcommand{\keywordsname}{Palabras clave}
     \renewcommand{\tablename}{Tabla}
     \renewcommand{\listtablename}{\'Indice de tablas}
     \renewcommand{\appendicesname}{Ap\'endices}}
     
\addto\captionsenglish{%
     \renewcommand{\titulacioname}{Degree in Mathematics}
     \renewcommand{\tfgname}{}
     \renewcommand{\authorname}{Author}
     \renewcommand{\tutorname}{Tutor}
     \renewcommand{\coursename}{Course}
     \renewcommand{\keywordsname}{Keywords}
     \renewcommand{\appendicesname}{Appendices}}
     




\RequirePackage{titlesec}

\titlelabel{\thetitle.\quad}

\titleformat{\chapter}[display]
{\normalfont\Large\filleft\bfseries}  % FORMAT
{%\titlerule[1pt]                                  
 %\vspace{1pt}
 %\titlerule
 %\vspace{1pc}
 \LARGE\MakeUppercase{\chaptertitlename} \thechapter} % LABEL
{0pc}                                                 % SEP
{\Huge}                                               % BEFORE CODE 
[\vspace{5pt}{\titlerule[1pt]}]                       % AFTER  CODE 

%\titleformat{\section}
% {\Large\normalfont%\sffamily
% \bfseries}
% {\thesection}{0.5em}{}[]%\hrule height1.5pt]
%


\RequirePackage{tocloft}
%
\setlength{\cftbeforechapskip}{3pt}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Definici\'on de las cabeceras y pies de p\'agina
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\RequirePackage{fancyhdr}

\pagestyle{fancy}
\setlength{\headheight}{16pt}
%\renewcommand*{\headheight}{14pt}
\renewcommand*{\headrulewidth}{0.7pt}

\renewcommand{\chaptermark}[1]{\markboth{{\sffamily\mdseries\normalsize\hspace{1em} #1}}{}}
\renewcommand{\sectionmark}[1]{\markright{{\sffamily\mdseries\normalsize\thesection\hspace{0.5em} #1}}{}}

\lhead[\bfseries\thepage]{\sffamily\rightmark}
\rhead[\sffamily\leftmark]{\bfseries\thepage}
\cfoot{} 
\lfoot{}
\rfoot{}


\RequirePackage{graphicx}
\RequirePackage{eso-pic}


%%%%%%%%%%%%%%
%  Navegable
%%%%%%%%%%%%%

\RequirePackage[colorlinks=true, linkcolor=blue,citecolor=blue,urlcolor=blue]{hyperref}



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Separaci\'on entre p\'arrafos
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\setlength{\parskip}{1ex}

\newcommand{\foreign}[2]{\foreignlanguage{#1}{\textit{#2}}}
     
\newenvironment{Foreign}[1]{\begin{quote}\itshape\selectlanguage{#1}}{\end{quote}}     
      
%%%%%%%%
% Resúmenes
%%%%%%%%


\newenvironment{abstract}[1][esp]{%
        \begingroup
          \selectlanguage{#1}\addvspace{\baselineskip}\par\hfill
          {\sffamily\bfseries\Large\abstractname}\smallskip\par}{%
          \par\nopagebreak\noindent
          \iflanguage{english}{\endgroup\makeindexes}{\endgroup\par\vspace{\stretch{1}}}}


%%%%%%%%%
% Índices
%%%%%%%%%
\newcommand{\makeindexes}{\cleardoublepage\tableofcontents\cleardoublepage
                          \iflistfigures\listoffigures\fi
                          \iflisttables\listoftables\fi
                          \addtocontents{toc}{\protect\noindent\protect\hrulefill\protect\par}
                          }

%\setlength{\cftaftertoctitleskip}{\baselineskip}
%\setlength{\cftafterloftitleskip}{\baselineskip}
%\setlength{\cftafterlottitleskip}{\baselineskip}
%
%\renewcommand{\cfttoctitlefont}{\hfill\Huge\sffamily\bfseries}
%\renewcommand{\cftlottitlefont}{\hfill\Huge\sffamily\bfseries}
%\renewcommand{\cftloftitlefont}{\hfill\Huge\sffamily\bfseries}
%
%\renewcommand{\cftaftertoctitle}{\par\noindent\hrulefill}
%\renewcommand{\cftafterlottitle}{\par\noindent\hrulefill}
%\renewcommand{\cftafterloftitle}{\par\noindent\hrulefill}
%
%\newcommand{\APPENDIX}{\appendix
%             \phantomsection
%             \addtocontents{toc}{\protect\noindent\protect\hrulefill\protect\par}             
%             \ifnum\value{appendix}=1 \addtocontents{toc}{\protect\appendixname\protect\par}
%                    \else\addtocontents{toc}{\protect\appendicesname\protect\par}\fi}

%%%% P\'agina de título

\newcommand{\m@ketitle}{%
\begin{titlepage}
\AddToShipoutPicture*{\AtPageUpperLeft{%
    % M\'argenes:  
    \raisebox{-4cm}{\hspace{2cm}\includegraphics[height=2.2cm]{logouam}% 
    \hspace{5.54cm}% 21cm-(2+6.27+5.19+2)cm
                    \includegraphics[height=2.2cm]{logodpto}% 
                    }}}
                    
\AddToShipoutPicture*{\includegraphics[width=21cm]{base}}
\centering
\vspace*{2cm}

\begin{color}{grisfosc}
\large
Departamento de Matem\'aticas, Facultad de Ciencias

Universidad Aut\'onoma de Madrid
\end{color}
\vspace{\stretch{1}}

\huge\bfseries\@title

\vspace*{1cm}
\large\scshape \tfgname
\par\vspace{0.2cm}\par
\normalfont{\titulacioname}

\vspace{\stretch{2}}

% Autor, tutor y fecha
\begin{flushright} 
\begin{tabular}{l}
   \textit{\authorname:} \begin{tabular}[t]{l}\@author\end{tabular}\\[0.2cm]
   \textit{\tutorname:} \begin{tabular}[t]{l}\t@tor\end{tabular}
\end{tabular}
\medskip

\coursename\space\c@rs
\end{flushright}
\end{titlepage}}


%%%%%%%%%%%%%%%%%
\renewcommand{\cleardoublepage}{\clearpage\if@twoside \ifodd\c@page\else
\hbox{}
\thispagestyle{empty}
\newpage
\if@twocolumn\hbox{}\newpage\fi\fi\fi}

\AtBeginDocument{%
    \frontmatter\m@ketitle\cleardoublepage\thispagestyle{plain}}

\newcounter{appendix}
\AtEndDocument{%
    \ifnum\value{chapter}=1 
           \immediate\write\@auxout{\unexpanded{\setcounter{appendix}{1}}}
    \else 
           \immediate\write\@auxout{\unexpanded{\setcounter{appendix}{2}}} \fi}
