import readline  # noqa: F401
import sys
import typer
from click import BadArgumentUsage, MissingParameter
from click.types import Choice

from sgpt.config import cfg
from sgpt.handlers.chat_handler import ChatHandler
from sgpt.handlers.default_handler import DefaultHandler
from sgpt.handlers.repl_handler import ReplHandler
from sgpt.role import DefaultRoles, SystemRole
from sgpt.utils import get_edited_prompt, install_shell_integration, run_command

def handle_second_ai(chat_handler, second_ai_prompt, model, temperature, top_probability, chat_id, caching):
    chat_handler.handle(
        second_ai_prompt,
        model=model,
        temperature=temperature,
        top_probability=top_probability,
        chat_id=chat_id,
        caching=caching,
    )

def handle_user_input(
    chat_handler: ChatHandler,
    user_input: str,
    model: str,
    temperature: float,
    top_probability: float,
    chat_id: str,
    caching: bool,
):
    user_response = chat_handler.handle(
        user_input,
        model=model,
        temperature=temperature,
        top_probability=top_probability,
        chat_id=chat_id,
        caching=caching,
    )
    print("You:", user_input)
    print("AI:", user_response)

def main(
    prompt: str = typer.Argument(
        None,
        show_default=False,
        help="The prompt to generate completions for.",
    ),
    model: str = typer.Option(
        cfg.get("DEFAULT_MODEL"),
        help="Large language model to use.",
    ),
    temperature: float = typer.Option(
        0.1,
        min=0.0,
        max=2.0,
        help="Randomness of generated output.",
    ),
    top_probability: float = typer.Option(
        1.0,
        min=0.1,
        max=1.0,
        help="Limits highest probable tokens (words).",
    ),
    shell: bool = typer.Option(
        False,
        "--shell",
        "-s",
        help="Generate and execute shell commands.",
        rich_help_panel="Assistance Options",
    ),
    describe_shell: bool = typer.Option(
        False,
        "--describe-shell",
        "-d",
        help="Describe a shell command.",
        rich_help_panel="Assistance Options",
    ),
    code: bool = typer.Option(
        False,
        help="Generate only code.",
        rich_help_panel="Assistance Options",
    ),
    editor: bool = typer.Option(
        False,
        help="Open $EDITOR to provide a prompt.",
    ),
    cache: bool = typer.Option(
        True,
        help="Cache completion results.",
    ),
    chat: str = typer.Option(
        None,
        help="Follow conversation with id, use 'temp' for quick session.",
        rich_help_panel="Chat Options",
    ),
    repl: str = typer.Option(
        None,
        help="Start a REPL (Read–eval–print loop) session.",
        rich_help_panel="Chat Options",
    ),
    show_chat: str = typer.Option(
        None,
        help="Show all messages from provided chat id.",
        callback=ChatHandler.show_messages_callback,
        rich_help_panel="Chat Options",
    ),
    list_chats: bool = typer.Option(
        False,
        help="List all existing chat ids.",
        callback=ChatHandler.list_ids,
        rich_help_panel="Chat Options",
    ),
    role: str = typer.Option(
        None,
        help="System role for GPT model.",
        rich_help_panel="Role Options",
    ),
    create_role: str = typer.Option(
        None,
        help="Create role.",
        callback=SystemRole.create,
        rich_help_panel="Role Options",
    ),
    show_role: str = typer.Option(
        None,
        help="Show role.",
        callback=SystemRole.show,
        rich_help_panel="Role Options",
    ),
    list_roles: bool = typer.Option(
        False,
        help="List roles.",
        callback=SystemRole.list,
        rich_help_panel="Role Options",
    ),
    install_integration: bool = typer.Option(
        False,
        help="Install shell integration (ZSH and Bash only)",
        callback=install_shell_integration,
        hidden=True,  # Hiding since should be used only once.
    ),
    second_ai: bool = typer.Option(
        False,
        help="Enable the second AI instance to receive prompts from the initial AI prompt.",
    ),
    second_ai_prompt: str = typer.Option(
        None,
        help="Second AI prompt (use with --second-ai to pass a prompt to the second AI instance).",
    ),
    interactive_mode: bool = typer.Option(
        False,
        "--interactive",
        help="Enter interactive mode to provide user input during conversation.",
    ),
) -> None:
    stdin_passed = not sys.stdin.isatty()

    if stdin_passed and not repl:
        prompt = f"{sys.stdin.read()}\n\n{prompt or ''}"

    if interactive_mode:
        chat_handler = ChatHandler(chat, SystemRole.get(role))

        while True:
            user_input = input("You: ")
            handle_user_input(
                chat_handler,
                user_input,
                model=model,
                temperature=temperature,
                top_probability=top_probability,
                chat_id=chat,
                caching=cache,
            )

    if not prompt and not editor and not repl:
        raise MissingParameter(param_hint="PROMPT", param_type="string")

    if sum((shell, describe_shell, code)) > 1:
        raise BadArgumentUsage(
            "Only one of --shell, --describe-shell, and --code options can be used at a time."
        )

    if chat and repl:
        raise BadArgumentUsage("--chat and --repl options cannot be used together.")

    if editor and stdin_passed:
        raise BadArgumentUsage("--editor option cannot be used with stdin input.")

    role_class = (
        DefaultRoles.check_get(shell, describe_shell, code)
        if not role
        else SystemRole.get(role)
    )

    if repl:
        # Will be in an infinite loop here until the user exits with Ctrl+C.
        ReplHandler(repl, role_class).handle(
            prompt,
            model=model,
            temperature=temperature,
            top_probability=top_probability,
            chat_id=repl,
            caching=cache,
        )

    if chat:
        if second_ai and second_ai_prompt is not None:
            chat_handler = ChatHandler(chat, role_class)
            handle_second_ai(
                chat_handler,
                second_ai_prompt,
                model,
                temperature,
                top_probability,
                chat,
                cache,
            )
        else:
            chat_handler = ChatHandler(chat, role_class)
            second_ai_handler = ChatHandler("second_ai", role_class)
            while True:
                # First AI instance
                first_ai_response = chat_handler.handle(
                    prompt,
                    model=model,
                    temperature=temperature,
                    top_probability=top_probability,
                    chat_id=chat,
                    caching=cache,
                )
                print("First AI:", first_ai_response)

                # Second AI instance
                second_ai_response = second_ai_handler.handle(
                    second_ai_prompt,
                    model=model,
                    temperature=temperature,
                    top_probability=top_probability,
                    chat_id="second_ai",
                    caching=cache,
                )
                print("Second AI:", second_ai_response)

                prompt = first_ai_response
                second_ai_prompt = second_ai_response
    else:
        full_completion = DefaultHandler(role_class).handle(
            prompt,
            model=model,
            temperature=temperature,
            top_probability=top_probability,
            caching=cache,
        )

    while shell and not stdin_passed:
        option = typer.prompt(
            text="[E]xecute, [D]escribe, [A]bort",
            type=Choice(("e", "d", "a", "y"), case_sensitive=False),
            default="e" if cfg.get("DEFAULT_EXECUTE_SHELL_CMD") == "true" else "a",
            show_choices=False,
            show_default=False,
        )
        if option in ("e", "y"):
            run_command(full_completion)
        elif option == "d":
            DefaultHandler(DefaultRoles.DESCRIBE_SHELL.get_role()).handle(
                full_completion,
                model=model,
                temperature=temperature,
                top_probability=top_probability,
                caching=cache,
            )
            continue

if __name__ == "__main__":
    typer.run(main)
