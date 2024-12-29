def __init__(self, 
             usage_tracker,
             performance_tracker,
             config_manager,
             model_manager=None):
    """
    Initialize the Gradio interface.
    
    Args:
        usage_tracker: Usage tracking system
        performance_tracker: Performance monitoring system
        config_manager: Configuration management system
        model_manager: Model management system (optional)
    """
    self.usage_tracker = usage_tracker
    self.performance_tracker = performance_tracker
    self.config_manager = config_manager
    self.model_manager = model_manager
    self.update_lock = threading.Lock()
    self.theme = gr.themes.Soft()

def create_interface(self) -> gr.Blocks:
    """
    Create the main Gradio interface.
    
    Returns:
        gr.Blocks: Gradio interface
    """
    with gr.Blocks(theme=self.theme) as interface:
        with gr.Tabs():
            self._create_chat_tab()
            self._create_performance_tab()
            self._create_config_tab()
            self._create_usage_tab()
        return interface

def _create_chat_tab(self) -> None:
    """Create the chat interface tab."""
    with gr.Tab("Chat"):
        with gr.Row():
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(
                    label="Conversation",
                    height=500
                )
                msg = gr.Textbox(
                    label="Message",
                    placeholder="Type your message here...",
                    lines=2
                )
                with gr.Row():
                    clear = gr.Button("Clear")
                    send = gr.Button("Send", variant="primary")
            
            with gr.Column(scale=1):
                with gr.Tab("System Stats"):
                    tokens_used = gr.Number(
                        label="Tokens Used",
                        value=0,
                        precision=0
                    )
                    tiredness = gr.Number(
                        label="Tiredness Score",
                        value=0,
                        precision=2
                    )
                    last_dream = gr.Textbox(
                        label="Last Dream State",
                        value="Never"
                    )
                    current_models = gr.JSON(
                        label="Active Models",
                        value=self.config_manager.get_default_config()
                    )
                dream_button = gr.Button(
                    "Trigger Dream State",
                    variant="secondary"
                )

        # Event handlers
        def process_message(message: str, history: List) -> tuple:
            """Process user message and update chat."""
            if not message.strip():
                raise gr.Error("Message cannot be empty")
                
            try:
                with self.update_lock:
                    if self.model_manager:
                        response = self.model_manager.process_message(message)
                    else:
                        response = "Model manager not initialized"
                    
                    new_history = history + [[message, response]]
                    return "", new_history
            except Exception as e:
                raise gr.Error(f"Error processing message: {str(e)}")

        def trigger_dream_state() -> Dict[str, Any]:
            """Trigger a dream state and update stats."""
            try:
                if self.model_manager:
                    dream_results = self.model_manager.enter_dream_state()
                    return {
                        tiredness: 0,
                        last_dream: datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        tokens_used: 0
                    }
                return {}
            except Exception as e:
                raise gr.Error(f"Error triggering dream state: {str(e)}")

        # Set up event handlers
        msg.submit(process_message, [msg, chatbot], [msg, chatbot])
        send.click(process_message, [msg, chatbot], [msg, chatbot])
        clear.click(lambda: None, None, chatbot, queue=False)
        dream_button.click(trigger_dream_state, 
                         outputs=[tiredness, last_dream, tokens_used])

def _create_performance_tab(self) -> None:
    """Create the performance monitoring tab."""
    with gr.Tab("Model Performance"):
        with gr.Row():
            with gr.Column():
                model_select = gr.Dropdown(
                    choices=self.get_available_models(),
                    label="Select Model",
                    multiselect=True
                )
                time_range = gr.Dropdown(
                    choices=[
                        "Last 100 requests",
                        "Last 24 hours",
                        "Last 7 days",
                        "All time"
                    ],
                    label="Time Range",
                    value="Last 100 requests"
                )
                compare_btn = gr.Button("Compare Performance")
            
            with gr.Column():
                performance_chart = gr.Plot(label="Performance Comparison")
                efficiency_chart = gr.Plot(label="Cost Efficiency")

        with gr.Row():
            performance_table = gr.DataFrame(
                headers=[
                    "Model",
                    "Avg Response Time",
                    "Token Efficiency",
                    "Error Rate",
                    "Cost per Token",
                    "Success Rate"
                ],
                label="Performance Metrics"
            )

        # Benchmarking section
        with gr.Row():
            gr.Markdown("## Benchmark Suite")
        
        with gr.Row():
            with gr.Column():
                benchmark_text = gr.TextArea(
                    label="Benchmark Prompts (One per line)",
                    placeholder="Enter test prompts..."
                )
                expected_outputs = gr.TextArea(
                    label="Expected Outputs (One per line)",
                    placeholder="Enter expected outputs..."
                )
                run_benchmark = gr.Button("Run Benchmark")
            
            with gr.Column():
                benchmark_results = gr.DataFrame(
                    headers=[
                        "Model",
                        "Accuracy",
                        "Avg Response Time",
                        "Cost Efficiency"
                    ],
                    label="Benchmark Results"
                )

        # Event handlers
        def update_performance_charts(selected_models: List[str],
                                   time_range: str) -> Dict[str, Any]:
            """Update performance comparison charts."""
            if not selected_models:
                raise gr.Error("Please select at least one model")
                
            try:
                gr.Info("Updating charts...")
                all_data = []
                
                for model in selected_models:
                    summary = self.performance_tracker.get_model_performance_summary(
                        model
                    )
                    if summary:
                        all_data.append({
                            "model": model,
                            **summary
                        })
                
                if not all_data:
                    raise gr.Error("No performance data available")
                
                df = pd.DataFrame(all_data)
                
                # Create performance comparison plot
                perf_fig = px.box(
                    df,
                    x="model",
                    y="average_response_time",
                    title="Response Time Distribution by Model"
                )
                
                # Create efficiency comparison plot
                eff_fig = px.scatter(
                    df,
                    x="token_efficiency",
                    y="average_cost_per_token",
                    color="model",
                    title="Cost Efficiency vs Token Efficiency"
                )
                
                # Format table data
                table_data = [[
                    model,
                    f"{data['average_response_time']:.3f}",
                    f"{data['token_efficiency']:.2f}",
                    f"{data['error_rate']*100:.1f}%",
                    f"${data['average_cost_per_token']:.5f}",
                    f"{data['success_rate']*100:.1f}%"
                ] for model, data in zip(df['model'], all_data)]
                
                return {
                    performance_chart: perf_fig,
                    efficiency_chart: eff_fig,
                    performance_table: table_data
                }
            
            except Exception as e:
                raise gr.Error(f"Error updating charts: {str(e)}")

        def run_benchmark_suite(selected_models: List[str],
                             prompts: str,
                             expected: str) -> pd.DataFrame:
            """Run benchmark tests on selected models."""
            try:
                prompts_list = [p.strip() for p in prompts.split('\n') if p.strip()]
                expected_list = [e.strip() for e in expected.split('\n') if e.strip()]
                
                if len(prompts_list) != len(expected_list):
                    raise gr.Error("Number of prompts and expected outputs must match")
                
                benchmark_suite = [
                    {"input": p, "expected_output": e}
                    for p, e in zip(prompts_list, expected_list)
                ]
                
                benchmark_id = self.performance_tracker.run_benchmark(
                    selected_models,
                    benchmark_suite
                )
                
                results = self.performance_tracker.get_benchmark_results(benchmark_id)
                if not results:
                    raise gr.Error("Benchmark failed to complete")
                
                return pd.DataFrame([
                    [
                        model,
                        f"{metrics['accuracy']*100:.1f}%",
                        f"{metrics['average_response_time']:.3f}s",
                        f"${metrics['cost_efficiency']:.5f}/token"
                    ]
                    for model, data in results.items()
                    for metrics in [data['metrics']]
                ], columns=benchmark_results.headers)
            
            except Exception as e:
                raise gr.Error(f"Error running benchmark: {str(e)}")

        # Set up event handlers
        compare_btn.click(
            update_performance_charts,
            inputs=[model_select, time_range],
            outputs=[performance_chart, efficiency_chart, performance_table]
        )
        
        run_benchmark.click(
            run_benchmark_suite,
            inputs=[model_select, benchmark_text, expected_outputs],
            outputs=[benchmark_results]
        )

def _create_config_tab(self) -> None:
    """Create the configuration management tab."""
    with gr.Tab("Configuration"):
        with gr.Row():
            with gr.Column():
                gr.Markdown("## Model Configuration")
                local_model = gr.Dropdown(
                    choices=self.get_available_models(),
                    label="Local Model",
                    value=self.config_manager.get_default_config()["local_model"]
                )
                cloud_model = gr.Dropdown(
                    choices=self.get_available_models(),
                    label="Cloud Model",
                    value=self.config_manager.get_default_config()["cloud_model"]
                )
                
                gr.Markdown("## API Keys")
                hf_api_key = gr.Textbox(
                    label="HuggingFace API Key",
                    type="password"
                )
                anthropic_api_key = gr.Textbox(
                    label="Anthropic API Key",
                    type="password"
                )
                openai_api_key = gr.Textbox(
                    label="OpenAI API Key",
                    type="password"
                )
            
            with gr.Column():
                gr.Markdown("## Presets")
                preset_name = gr.Textbox(label="Preset Name")
                save_preset = gr.Button("Save as Preset")
                preset_list = gr.Dropdown(
                    choices=self.config_manager.list_presets(),
                    label="Load Preset"
                )
                load_preset = gr.Button("Load Selected Preset")

        # Event handlers
        def save_preset_config(name: str,
                            local_mod: str,
                            cloud_mod: str,
                            hf_key: str,
                            anthropic_key: str,
                            openai_key: str) -> None:
            """Save current configuration as preset."""
            try:
                config = {
                    "local_model": local_mod,
                    "cloud_model": cloud_mod,
                    "hf_api_key": hf_key,
                    "anthropic_api_key": anthropic_key,
                    "openai_api_key": openai_key
                }
                
                if self.config_manager.save_preset(config, name):
                    gr.Info(f"Preset '{name}' saved successfully")
                else:
                    raise gr.Error("Failed to save preset")
            
            except Exception as e:
                raise gr.Error(f"Error saving preset: {str(e)}")

        def load_preset_config(preset_name: str) -> Dict[str, str]:
            """Load selected preset configuration."""
            try:
                config = self.config_manager.get_preset(preset_name)
                if not config:
                    raise gr.Error(f"Preset '{preset_name}' not found")
                
                return {
                    local_model: config["local_model"],
                    cloud_model: config["cloud_model"],
                    hf_api_key: config["hf_api_key"],
                    anthropic_api_key: config["anthropic_api_key"],
                    openai_api_key: config["openai_api_key"]
                }
            
            except Exception as e:
                raise gr.Error(f"Error loading preset: {str(e)}")

        # Set up event handlers
        save_preset.click(
            save_preset_config,
            inputs=[preset_name, local_model, cloud_model,
                   hf_api_key, anthropic_api_key, openai_api_key]
        )
        
        load_preset.click(
            load_preset_config,
            inputs=[preset_list],
            outputs=[local_model, cloud_model,
                    hf_api_key, anthropic_api_key, openai_api_key]
        )

def _create_usage_tab(self) -> None:
    """Create the usage monitoring tab."""
    with gr.Tab("Usage & Costs"):
        with gr.Row():
            with gr.Column():
                time_period = gr.Dropdown(
                    choices=["7 days", "30 days", "90 days", "All time"],
                    label="Time Period",
                    value="7 days"
                )
                refresh_btn = gr.Button("Refresh Stats")
            
            with gr.Column():
                total_cost = gr.Number(
                    label="Total Cost ($)",
                    precision=4
                )
                total_tokens = gr.Number(
                    label="Total Tokens Used"
                )
        
        with gr.Row():
            cost_plot = gr.Plot(label="Cost Trends")