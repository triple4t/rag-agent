import { User, Bot } from 'lucide-react';

interface AvatarProps {
  type: 'user' | 'assistant';
}

const Avatar = ({ type }: AvatarProps) => {
  return (
    <div
      className={`w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 ${
        type === 'user' ? 'bg-avatar-user' : 'bg-avatar-assistant'
      }`}
    >
      {type === 'user' ? (
        <User className="w-5 h-5 text-foreground" />
      ) : (
        <Bot className="w-5 h-5 text-foreground" />
      )}
    </div>
  );
};

export default Avatar;
