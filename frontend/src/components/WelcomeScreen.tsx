import { FileText, MessageSquare, Search, Zap } from 'lucide-react';

interface WelcomeScreenProps {
  hasDocuments: boolean;
  onSuggestionClick: (suggestion: string) => void;
}

const suggestions = [
  {
    icon: Search,
    text: "What are the main topics covered in my documents?",
  },
  {
    icon: MessageSquare,
    text: "Summarize the key points from the uploaded files",
  },
  {
    icon: FileText,
    text: "Find specific information about...",
  },
  {
    icon: Zap,
    text: "Compare and contrast ideas across documents",
  },
];

const WelcomeScreen = ({ hasDocuments, onSuggestionClick }: WelcomeScreenProps) => {
  return (
    <div className="flex-1 flex flex-col items-center justify-center px-4 py-12">
      <h1 className="text-4xl font-semibold text-foreground mb-8 text-center">
        How can I help you today?
      </h1>

      {hasDocuments ? (
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 max-w-2xl w-full">
          {suggestions.map((suggestion, index) => (
            <button
              key={index}
              onClick={() => onSuggestionClick(suggestion.text)}
              className="flex items-start gap-3 p-4 rounded-xl border border-border bg-card hover:bg-accent transition-colors text-left group"
            >
              <suggestion.icon className="w-5 h-5 text-muted-foreground group-hover:text-foreground transition-colors flex-shrink-0 mt-0.5" />
              <span className="text-sm text-muted-foreground group-hover:text-foreground transition-colors">
                {suggestion.text}
              </span>
            </button>
          ))}
        </div>
      ) : (
        <div className="text-center">
          <p className="text-muted-foreground mb-2">
            Upload documents to get started
          </p>
          <p className="text-sm text-muted-foreground/70">
            Click the sidebar to upload PDF files, then ask questions about them
          </p>
        </div>
      )}
    </div>
  );
};

export default WelcomeScreen;
