const TypingIndicator = () => {
  return (
    <div className="flex items-center gap-1 px-4 py-2">
      <div className="w-2 h-2 rounded-full bg-muted-foreground typing-dot" />
      <div className="w-2 h-2 rounded-full bg-muted-foreground typing-dot" />
      <div className="w-2 h-2 rounded-full bg-muted-foreground typing-dot" />
    </div>
  );
};

export default TypingIndicator;
