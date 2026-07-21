Name:           agenttool
Version:        0.2.0
Release:        1%{?dist}
Summary:        Terminal-based agentic coding assistant powered by local LLMs
License:        Apache-2.0
URL:            https://github.com/aditya-an1l/AgentTool
Source0:        %{url}/archive/v%{version}/%{name}-%{version}.tar.gz
BuildArch:      noarch
BuildRequires:  python3-devel
BuildRequires:  python3-setuptools
Requires:       python3-openai
Requires:       python3-ddgs
Requires:       python3-rich
Requires:       python3-prompt-toolkit
Requires:       python3-requests

%description
AgentTool is a lightweight Python CLI that discovers locally-running LLM
servers (Ollama or LM Studio), lets you pick a model, and runs an
OpenAI-compatible agentic loop with tool-calling capabilities. It supports
web search, file I/O, directory listing, and shell command execution.

%prep
%autosetup -n %{name}-%{version}

%build
%py3_build

%install
%py3_install

%files
%license LICENSE
%doc README.md
%{_bindir}/agenttool
%{python3_sitelib}/%{name}/

%changelog
* Tue Jul 21 2026 Aditya Anil <aditya.anil.productions@gmail.com> - 0.2.0-1
- Initial RPM release
